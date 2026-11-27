"""Estimate likely ball ownership from observed image-space tracks."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from pitchlens.ball_contact import Contact, Point, TeamId, TrackKey, nearest_contact, observed_ball
from pitchlens.video_timing import FrameClock, is_finite_number


@dataclass(frozen=True)
class PossessionConfig:
    max_normalized_distance: float = 0.30
    min_distance_margin: float = 0.1
    confirmation_seconds: float = 0.1
    hysteresis_seconds: float = 0.12
    max_relative_speed: float = 1.5
    max_observation_gap_seconds: float = 0.25

    def __post_init__(self):
        for name in ('max_normalized_distance', 'confirmation_seconds', 'max_relative_speed', 'max_observation_gap_seconds'):
            value = getattr(self, name)
            if not is_finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        margin = self.min_distance_margin
        if not is_finite_number(margin) or not 0 <= margin <= self.max_normalized_distance:
            raise ValueError('min_distance_margin must be between zero and max_normalized_distance')
        if not is_finite_number(self.hysteresis_seconds) or self.hysteresis_seconds < 0:
            raise ValueError('hysteresis_seconds must be nonnegative and finite')


@dataclass
class _TrackPossession:
    duration_seconds: float = 0.
    episodes: int = 0
    known_teams: set[TeamId] = field(default_factory=set)


class PossessionAnalytics:
    """Record frame estimates and account for each measured interval once."""

    def __init__(self, fps: float, team_ids: tuple[TeamId, TeamId], config: PossessionConfig | None = None):
        if not is_finite_number(fps) or fps <= 0:
            raise ValueError('fps must be positive and finite')
        if (len(team_ids) != 2 or any(type(team) not in (int, str) for team in team_ids)
                or team_ids[0] == team_ids[1]):
            raise ValueError('Expected two distinct team identities')
        self.config = config or PossessionConfig()
        self._clock = FrameClock(fps)
        self._teams = dict.fromkeys(team_ids, 0.)
        self._tracks: dict[TrackKey, _TrackPossession] = {}
        self._segment: int | None = None
        self._candidate: Contact | None = None
        self._support_origin: Point | None = None
        self._consistent_seconds = 0.
        self._confirmed: Contact | None = None
        self._held: TrackKey | None = None
        self._unsupported_seconds = 0.
        self._analyzed_seconds = 0.
        self._player_seconds = 0.

    def update(self, frame_index: int, segment: int, camera_cut: bool, tracks: Sequence[dict], *,
               timestamp_seconds: float | None = None, ambiguous_ball: bool = False) -> dict:
        if self._clock.frame_index is not None and frame_index <= self._clock.frame_index:
            elapsed, source, valid_clock = None, None, False
        else:
            elapsed, source, valid_clock = self._clock.step(frame_index, timestamp_seconds)
        config = self.config
        continuous = (not camera_cut and segment == self._segment and elapsed is not None
                      and elapsed <= config.max_observation_gap_seconds + 1e-9)
        previous = self._confirmed if continuous else None
        if not continuous:
            self._held = None
            self._unsupported_seconds = 0.
        ball, ball_status, reason = observed_ball(tracks, ambiguous_ball)
        candidate = None
        if ball is not None:
            candidate, reason = nearest_contact(segment, tracks, ball, self._teams,
                                                config.max_normalized_distance, config.min_distance_margin,
                                                lower_region=True)
        if not valid_clock:
            candidate, reason = None, 'invalid_timing'

        moving = False
        if continuous and candidate is not None and self._candidate is not None and candidate.key == self._candidate.key:
            # Measure movement relative to the person, so a shared pan cancels out.
            # Keep a small normalized slack for detector and box jitter.
            duration = self._consistent_seconds + elapsed
            offset = candidate.relative_offset
            moving = math.dist(offset, self._candidate.relative_offset) > config.max_relative_speed * elapsed + .025
            if previous is None and self._support_origin is not None:
                moving |= math.dist(offset, self._support_origin) > config.max_relative_speed * duration + .025
            self._consistent_seconds = 0. if moving else duration
        else:
            self._consistent_seconds = 0.
        if not self._consistent_seconds:
            self._support_origin = candidate.relative_offset if candidate is not None else None
        self._segment = segment
        self._candidate = candidate
        # A hold never supplies evidence or time credit. It only lets the same
        # owner resume on a fresh observed contact after a brief interruption.
        if continuous and self._held is not None:
            self._unsupported_seconds += elapsed
            if previous is None and self._unsupported_seconds > config.hysteresis_seconds + 1e-9:
                self._held = None
        supported = candidate if not moving else None
        if moving:
            reason = 'relative_ball_travel'
        held = self._held
        self._confirmed = None
        if supported is not None and (supported.key == held or
                                      self._consistent_seconds >= config.confirmation_seconds - 1e-9):
            self._confirmed = supported
            self._held = supported.key
            self._unsupported_seconds = 0.

        credited_seconds, credited_team = 0., None
        if elapsed is not None:
            self._analyzed_seconds += elapsed
        if self._confirmed is not None:
            track = self._tracks.get(supported.key)
            if track is None:
                track = self._tracks[supported.key] = _TrackPossession()
            if supported.team is not None:
                track.known_teams.add(supported.team)
            if held != supported.key:
                track.episodes += 1
            if previous is not None and previous.key == supported.key:
                # Credit only intervals with the same confirmed owner at both ends.
                credited_seconds = elapsed
                track.duration_seconds += elapsed
                self._player_seconds += elapsed
                if supported.team is not None and previous.team == supported.team:
                    credited_team = supported.team
                    self._teams[credited_team] += elapsed

        state = 'confirmed' if self._confirmed is not None else 'pending' if supported is not None else 'unknown'
        return {
            'state': state, 'reason': 'confirmation_pending' if state == 'pending' else reason,
            'owner': self._confirmed.identity() if self._confirmed is not None else None,
            'pending_candidate': supported.identity() if state == 'pending' else None,
            'normalized_distance': candidate.distance if candidate is not None else None,
            'ball_status': ball_status,
            'timing': {'elapsed_seconds': elapsed, 'source': source,
                       'credited_player_seconds': credited_seconds, 'credited_team': credited_team},
            'totals': self.totals(),
        }

    def totals(self) -> dict:
        analyzed = self._analyzed_seconds
        known = sum(self._teams.values())
        return {
            'analyzed_duration_seconds': analyzed,
            'confirmed_player_duration_seconds': self._player_seconds,
            'known_team_duration_seconds': known,
            'unknown_player_duration_seconds': max(0., analyzed - self._player_seconds),
            'player_with_unknown_team_duration_seconds': max(0., self._player_seconds - known),
            'unknown_team_duration_seconds': max(0., analyzed - known),
            'player_possession_coverage_pct': 100 * self._player_seconds / analyzed if analyzed else None,
            'team_possession_coverage_pct': 100 * known / analyzed if analyzed else None,
            'teams': [{'team': team, 'duration_seconds': seconds,
                       'possession_pct': 100 * seconds / known if known else None}
                      for team, seconds in self._teams.items()],
        }

    def summary(self) -> dict:
        current = self._confirmed.key if self._confirmed is not None else None
        tracks = []
        for (segment, class_id, track_id), track in sorted(self._tracks.items()):
            team = next(iter(track.known_teams)) if len(track.known_teams) == 1 else None
            tracks.append({'segment': segment, 'class_id': class_id, 'track_id': track_id, 'team': team,
                           'duration_seconds': track.duration_seconds, 'episodes': track.episodes,
                           'status': 'confirmed' if current == (segment, class_id, track_id) else 'inactive'})
        return {**self.totals(), 'tracks': tracks, 'episodes': sum(track['episodes'] for track in tracks)}
