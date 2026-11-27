"""Estimate likely ball ownership from observed image-space tracks."""

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from pitchlens.ball_contact import Contact, TeamId, TrackKey, nearest_contact, observed_ball
from pitchlens.video_timing import FrameClock, is_finite_number


@dataclass(frozen=True)
class PossessionConfig:
    max_normalized_distance: float = 0.5
    min_distance_margin: float = 0.1
    confirmation_seconds: float = 0.1

    def __post_init__(self):
        for name in ('max_normalized_distance', 'confirmation_seconds'):
            value = getattr(self, name)
            if not is_finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        margin = self.min_distance_margin
        if not is_finite_number(margin) or not 0 <= margin <= self.max_normalized_distance:
            raise ValueError('min_distance_margin must be between zero and max_normalized_distance')


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
        self._consistent_seconds = 0.
        self._confirmed = False
        self._analyzed_seconds = 0.
        self._player_seconds = 0.

    def update(self, frame_index: int, segment: int, camera_cut: bool, tracks: Sequence[dict], *,
               timestamp_seconds: float | None = None, ambiguous_ball: bool = False) -> dict:
        if self._clock.frame_index is not None and frame_index <= self._clock.frame_index:
            elapsed, source, valid_clock = None, None, False
        else:
            elapsed, source, valid_clock = self._clock.step(frame_index, timestamp_seconds)
        continuous = not camera_cut and segment == self._segment and elapsed is not None
        previous = self._candidate if continuous and self._confirmed else None
        ball, ball_status, reason = observed_ball(tracks, ambiguous_ball)
        candidate = None
        if ball is not None:
            candidate, reason = nearest_contact(segment, tracks, ball, self._teams,
                                                self.config.max_normalized_distance, self.config.min_distance_margin)
        if not valid_clock:
            candidate, reason = None, 'invalid_timing'

        if (continuous and candidate is not None and self._candidate is not None
                and candidate.key == self._candidate.key):
            self._consistent_seconds += elapsed
        else:
            self._consistent_seconds = 0.
        self._segment = segment
        self._candidate = candidate
        threshold = self.config.confirmation_seconds
        self._confirmed = candidate is not None and (
            self._consistent_seconds >= threshold or math.isclose(self._consistent_seconds, threshold, rel_tol=1e-9)
        )

        credited_seconds, credited_team = 0., None
        if elapsed is not None:
            self._analyzed_seconds += elapsed
        if self._confirmed:
            track = self._tracks.get(candidate.key)
            if track is None:
                track = self._tracks[candidate.key] = _TrackPossession()
            if candidate.team is not None:
                track.known_teams.add(candidate.team)
            if previous is None or previous.key != candidate.key:
                track.episodes += 1
            else:
                # Credit only intervals with the same confirmed owner at both ends.
                credited_seconds = elapsed
                track.duration_seconds += elapsed
                self._player_seconds += elapsed
                if candidate.team is not None and previous.team == candidate.team:
                    credited_team = candidate.team
                    self._teams[credited_team] += elapsed

        state = 'confirmed' if self._confirmed else 'pending' if candidate is not None else 'unknown'
        return {
            'state': state, 'reason': 'confirmation_pending' if state == 'pending' else reason,
            'owner': candidate.identity() if self._confirmed else None,
            'pending_candidate': candidate.identity() if state == 'pending' else None,
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
        current = self._candidate.key if self._confirmed else None
        tracks = []
        for (segment, class_id, track_id), track in sorted(self._tracks.items()):
            team = next(iter(track.known_teams)) if len(track.known_teams) == 1 else None
            tracks.append({'segment': segment, 'class_id': class_id, 'track_id': track_id, 'team': team,
                           'duration_seconds': track.duration_seconds, 'episodes': track.episodes,
                           'status': 'confirmed' if current == (segment, class_id, track_id) else 'inactive'})
        return {**self.totals(), 'tracks': tracks, 'episodes': sum(track['episodes'] for track in tracks)}
