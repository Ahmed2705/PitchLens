"""Detect likely ball transfers using observed contact and image-space travel."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from pitchlens.ball_contact import (
    Box, Contact, Point, TeamId, TrackKey, nearest_contact, observed_ball, track_box,
)
from pitchlens.video_timing import FrameClock, is_finite_number


@dataclass(frozen=True)
class PassConfig:
    max_normalized_contact_distance: float = 0.5
    min_contact_margin: float = 0.1
    min_contact_seconds: float = 0.05
    min_normalized_travel: float = 0.75
    min_transfer_seconds: float = 0.08
    max_transfer_seconds: float = 3.0
    max_observation_gap_seconds: float = 0.2
    min_travel_samples: int = 2

    def __post_init__(self):
        for name in ('max_normalized_contact_distance', 'min_contact_seconds', 'min_normalized_travel',
                     'min_transfer_seconds', 'max_transfer_seconds', 'max_observation_gap_seconds'):
            value = getattr(self, name)
            if not is_finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not is_finite_number(self.min_contact_margin) or not 0 <= self.min_contact_margin <= self.max_normalized_contact_distance:
            raise ValueError('min_contact_margin must be between zero and the contact distance limit')
        if (self.min_transfer_seconds > self.max_transfer_seconds
                or self.max_observation_gap_seconds > self.max_transfer_seconds):
            raise ValueError('Transfer duration bounds must accommodate the minimum duration and observation gap')
        if type(self.min_travel_samples) is not int or self.min_travel_samples < 2:
            raise ValueError('min_travel_samples must be an integer of at least 2')


@dataclass(frozen=True)
class _BallSample:
    frame: int
    seconds: float
    xy: Point


def _relative(ball: Point, box: Box) -> Point:
    x1, y1, x2, y2 = box
    height = y2 - y1
    return (ball[0] - (x1 + x2) / 2) / height, (ball[1] - y2) / height


@dataclass
class _ContactRun:
    first: _BallSample
    last: _BallSample
    contact: Contact
    box: Box
    first_contact: Contact
    teams: set[TeamId] = field(default_factory=set)
    supported: bool = False

    def update(self, sample: _BallSample, contact: Contact, box: Box, confirmed: bool, minimum: float):
        self.last, self.contact, self.box = sample, contact, box
        if contact.team is not None:
            self.teams.add(contact.team)
        duration = sample.seconds - self.first.seconds
        self.supported |= confirmed or duration >= minimum or math.isclose(duration, minimum, rel_tol=1e-9)

    def team_at_endpoint(self, *, at_reception: bool = False) -> TeamId | None:
        if len(self.teams) != 1 or self.contact.team is None:
            return None
        return self.first_contact.team if at_reception else self.contact.team


@dataclass
class _Flight:
    source: _ContactRun
    travel: list[tuple[_BallSample, Point]]
    last_observed: _BallSample
    max_gap: float
    missing: bool = False
    bridged_gap: bool = False
    receiver: _ContactRun | None = None
    relative_travel: float = 0.
    image_travel: float = 0.
    observed_relative_motion: float = 0.
    observed_image_motion: float = 0.


def _event_counts() -> dict[str, int]:
    return dict.fromkeys(('detected_passes', 'received_passes', 'possible_turnovers_from',
                          'possible_turnovers_to', 'unclassified_transfers_from',
                          'unclassified_transfers_to'), 0)


class PassDetector:
    """Track source contact, separated flight and reception within a single shot."""

    def __init__(self, fps: float, team_ids: tuple[TeamId, TeamId], config: PassConfig | None = None):
        if not is_finite_number(fps) or fps <= 0:
            raise ValueError('fps must be positive and finite')
        if len(team_ids) != 2 or team_ids[0] == team_ids[1] or any(type(team) not in (str, int) for team in team_ids):
            raise ValueError('Expected two distinct team identities')
        self.config = config or PassConfig()
        self._clock = FrameClock(fps)
        self._time = 0.
        self._segment: int | None = None
        self._contact: _ContactRun | None = None
        self._flight: _Flight | None = None
        self._teams = {team: _event_counts() for team in team_ids}
        self._players: dict[TrackKey, dict[str, int]] = {}
        self._events: list[dict] = []
        self._counts: Counter[str] = Counter()
        self._rejections: Counter[str] = Counter()
        self._latest_rejection: str | None = None
        self._frames = self._observed_frames = self._usable_frames = 0
        self._analyzed_seconds = 0.

    def _reject(self, reason: str):
        if self._flight is not None:
            self._rejections[reason] += 1
            self._latest_rejection = reason
        self._flight = None
        self._contact = None

    def _run(self, run: _ContactRun | None, sample: _BallSample, contact: Contact, box: Box,
             confirmed: bool) -> _ContactRun:
        if run is None or run.contact.key != contact.key:
            run = _ContactRun(sample, sample, contact, box, contact)
        run.update(sample, contact, box, confirmed, self.config.min_contact_seconds)
        return run

    def update(self, frame_index: int, segment: int, camera_cut: bool, tracks: Sequence[dict], *,
               timestamp_seconds: float | None = None, ambiguous_ball: bool = False,
               possession: dict | None = None) -> dict:
        self._latest_rejection = None
        if self._clock.frame_index is not None and frame_index <= self._clock.frame_index:
            self._reject('invalid_timing')
            return self._frame_output(None)
        elapsed, _, valid_clock = self._clock.step(frame_index, timestamp_seconds)
        self._frames += 1
        if any(t['class_id'] == 3 and t.get('observed') is True for t in tracks):
            self._observed_frames += 1
        if elapsed is None:
            self._time = (timestamp_seconds if valid_clock and self._clock.has_positive_timestamp
                          and timestamp_seconds is not None else frame_index / self._clock.fps)
        else:
            self._time += elapsed
            self._analyzed_seconds += elapsed
        if camera_cut or segment != self._segment:
            self._reject('camera_cut')
        elif elapsed is None:
            self._reject('invalid_timing' if not valid_clock else 'frame_or_timing_gap')
        elif elapsed > self.config.max_observation_gap_seconds + 1e-9:
            self._reject('observation_gap')
        self._segment = segment
        ball, _, reason = observed_ball(tracks, ambiguous_ball)
        self._usable_frames += ball is not None
        if not valid_clock:
            return self._frame_output(None)

        candidate = None
        if ball is not None:
            candidate, reason = nearest_contact(
                segment, tracks, ball, self._teams, self.config.max_normalized_contact_distance,
                self.config.min_contact_margin,
            )
        owner = possession.get('owner') if possession is not None and possession.get('state') == 'confirmed' else None
        confirmed = (candidate is not None and owner is not None
                     and (owner['segment'], owner['class_id'], owner['track_id']) == candidate.key)
        sample = _BallSample(frame_index, self._time, ball) if ball is not None else None
        event = self._advance(segment, tracks, sample, candidate, reason, confirmed)
        return self._frame_output(event)

    def _advance(self, segment: int, tracks: Sequence[dict], sample: _BallSample | None,
                 candidate: Contact | None, reason: str | None, confirmed: bool) -> dict | None:
        players = {}
        for item in tracks:
            track_id, class_id = item['track_id'], item['class_id']
            if item.get('observed') is True and type(track_id) is int and class_id in (0, 1):
                box = track_box(item)
                if box is not None:
                    team = item.get('team')
                    team = team if type(team) in (int, str) and team in self._teams else None
                    players[segment, class_id, track_id] = box, team

        if self._flight is not None:
            event = self._continue_flight(players, sample, candidate, reason, confirmed)
            if self._flight is not None or event is not None:
                return event

        if candidate is not None:
            box, _ = players[candidate.key]
            self._contact = self._run(self._contact, sample, candidate, box, confirmed)
            return None

        source = self._contact
        self._contact = None
        if source is not None and source.supported and sample is not None and reason == 'no_nearby_player':
            position = players.get(source.contact.key)
            gap = sample.seconds - source.last.seconds
            if position is not None and gap <= self.config.max_observation_gap_seconds + 1e-9:
                box, _ = position
                relative = _relative(sample.xy, box)
                if math.hypot(*relative) > self.config.max_normalized_contact_distance:
                    self._flight = _Flight(source, [(sample, relative)], sample, gap)
        return None

    def _continue_flight(self, players: dict[TrackKey, tuple[Box, TeamId | None]], sample: _BallSample | None,
                         contact: Contact | None, reason: str | None, confirmed: bool) -> dict | None:
        flight = self._flight
        config = self.config
        release, _ = flight.travel[0]
        source = players.get(flight.source.contact.key)
        gap = self._time - flight.last_observed.seconds
        if source is None:
            self._reject('source_track_lost')
        elif flight.receiver is None and self._time - release.seconds > config.max_transfer_seconds + 1e-9:
            self._reject('transfer_timeout')
        elif gap > config.max_observation_gap_seconds + 1e-9:
            self._reject('observation_gap')
        elif reason in ('ambiguous_ball', 'invalid_ball', 'ambiguous_owner'):
            self._reject(reason)
        elif flight.receiver is not None and (contact is None or contact.key != flight.receiver.contact.key):
            self._reject('receiver_contact_lost')
        elif sample is None:
            flight.missing = True
        else:
            box, source_team = source
            if source_team is not None:
                flight.source.teams.add(source_team)
            flight.last_observed = sample
            if flight.receiver is None:
                flight.max_gap = max(flight.max_gap, gap)
            relative = _relative(sample.xy, box)
            if contact is None:
                # A short gap must be bracketed by actual travel samples, not contact.
                if not flight.missing:
                    previous, previous_relative = flight.travel[-1]
                    height = flight.source.box[3] - flight.source.box[1]
                    flight.observed_relative_motion += math.dist(relative, previous_relative)
                    flight.observed_image_motion += math.dist(sample.xy, previous.xy) / height
                flight.bridged_gap |= flight.missing
                flight.missing = False
                flight.travel.append((sample, relative))
            elif contact.key == flight.source.contact.key:
                self._reject('source_recontact')
            elif flight.missing:
                self._reject('observation_gap_at_reception')
            elif flight.receiver is None:
                source_height = flight.source.box[3] - flight.source.box[1]
                anchor_relative = _relative(flight.source.last.xy, flight.source.box)
                flight.relative_travel = math.dist(anchor_relative, relative)
                flight.image_travel = math.dist(flight.source.last.xy, sample.xy) / source_height
                if sample.seconds - release.seconds < config.min_transfer_seconds - 1e-9:
                    self._reject('short_transfer')
                elif len(flight.travel) < config.min_travel_samples:
                    self._reject('insufficient_travel_samples')
                elif (min(flight.relative_travel, flight.image_travel) < config.min_normalized_travel
                      or min(flight.observed_relative_motion, flight.observed_image_motion)
                      < config.min_normalized_travel / 4):
                    self._reject('insufficient_travel')
            if self._flight is not None and contact is not None:
                receiver_box, _ = players[contact.key]
                flight.receiver = self._run(flight.receiver, sample, contact, receiver_box, confirmed)
                if flight.receiver.supported:
                    return self._complete(flight)
        return None

    def _complete(self, flight: _Flight) -> dict:
        source, receiver = flight.source, flight.receiver
        source_team = source.team_at_endpoint()
        receiver_team = receiver.team_at_endpoint(at_reception=True)
        if source_team is None or receiver_team is None:
            classification, fields = 'unclassified_transfer', ('unclassified_transfers_from', 'unclassified_transfers_to')
        elif source_team == receiver_team:
            classification, fields = 'likely_successful_pass', ('detected_passes', 'received_passes')
        else:
            classification, fields = 'possible_turnover', ('possible_turnovers_from', 'possible_turnovers_to')
        for run, team, field_name in zip((source, receiver), (source_team, receiver_team), fields):
            key = run.contact.key
            if key not in self._players:
                self._players[key] = _event_counts()
            self._players[key][field_name] += 1
            if team is not None:
                self._teams[team][field_name] += 1
        release = flight.travel[0][0]
        reception = receiver.first
        event = {
            'event_id': len(self._events) + 1, 'classification': classification, 'segment': source.contact.key[0],
            'source': {**source.contact.identity(), 'team': source_team},
            'receiver': {**receiver.contact.identity(), 'team': receiver_team},
            'release_frame': release.frame, 'reception_frame': reception.frame,
            'confirmation_frame': receiver.last.frame,
            'release_timestamp_seconds': release.seconds, 'reception_timestamp_seconds': reception.seconds,
            'duration_seconds': reception.seconds - release.seconds,
            'observed_travel_samples': len(flight.travel), 'max_observation_gap_seconds': flight.max_gap,
            'evidence': {
                'source_contact_frame': source.last.frame, 'source_contact_distance': source.contact.distance,
                'reception_contact_distance': receiver.first_contact.distance,
                'source_relative_travel': flight.relative_travel, 'image_travel': flight.image_travel,
                'observed_relative_motion': flight.observed_relative_motion,
                'observed_image_motion': flight.observed_image_motion,
                'bridged_missing_ball': flight.bridged_gap,
                'observed_trajectory': [{'frame_index': point.frame, 'xy_px': list(point.xy)}
                                        for point in [source.last, *(mid for mid, _ in flight.travel), reception]],
            },
        }
        self._events.append(event)
        self._counts[classification] += 1
        self._contact = receiver
        self._flight = None
        return event

    def _frame_output(self, event: dict | None) -> dict:
        return {'events': [event] if event is not None else [], 'rejection_reason': self._latest_rejection,
                'running': self.totals()}

    def totals(self) -> dict:
        return {
            'detected_transfers': len(self._events), 'likely_successful_passes': self._counts['likely_successful_pass'],
            'possible_turnovers': self._counts['possible_turnover'],
            'unclassified_transfers': self._counts['unclassified_transfer'],
            'rejected_candidates': sum(self._rejections.values()), 'frames': self._frames,
            'observed_ball_frames': self._observed_frames, 'usable_ball_frames': self._usable_frames,
            'observed_ball_coverage_pct': 100 * self._observed_frames / self._frames if self._frames else None,
            'analyzed_duration_seconds': self._analyzed_seconds,
            'teams': [{'team': team, **counts} for team, counts in self._teams.items()],
        }

    def finish(self) -> dict:
        """Close an unfinished flight and produce the final event and track lists."""
        self._reject('clip_end')
        return {**self.totals(), 'rejection_reasons': dict(self._rejections), 'events': self._events.copy(),
                'tracks': [{'segment': segment, 'class_id': class_id, 'track_id': track_id, **counts}
                           for (segment, class_id, track_id), counts in sorted(self._players.items())]}
