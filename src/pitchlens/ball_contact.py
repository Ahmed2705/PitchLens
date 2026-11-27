"""Shared observed-ball and image-space player-contact geometry."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
import math

from pitchlens.video_timing import is_finite_number


TeamId = int | str
TrackKey = tuple[int, int, int]
Point = tuple[float, float]
Box = tuple[float, float, float, float]


def track_box(track: dict) -> Box | None:
    box = track.get('xyxy')
    if not isinstance(box, (tuple, list)) or len(box) != 4 or not all(map(is_finite_number, box)):
        return None
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y2) if 0 <= x1 < x2 and 0 <= y1 < y2 else None


@dataclass(frozen=True)
class Contact:
    key: TrackKey
    team: TeamId | None
    distance: float
    relative_offset: Point = (0., 0.)

    def identity(self) -> dict:
        segment, class_id, track_id = self.key
        return {'segment': segment, 'class_id': class_id, 'track_id': track_id, 'team': self.team}


def observed_ball(tracks: Sequence[dict], ambiguous_ball: bool) -> tuple[Point | None, str, str | None]:
    balls = [track for track in tracks if track['class_id'] == 3]
    if ambiguous_ball or len(balls) > 1:
        return None, 'ambiguous', 'ambiguous_ball'
    if not balls:
        return None, 'missing', 'missing_ball'
    ball = balls[0]
    if ball.get('observed') is not True:
        return None, 'estimated', 'unobserved_ball'
    ball_box = track_box(ball)
    if ball_box is None:
        return None, 'invalid', 'invalid_ball'
    bx1, by1, bx2, by2 = ball_box
    return ((bx1 + bx2) / 2, (by1 + by2) / 2), 'observed', None


def nearest_contact(segment: int, tracks: Sequence[dict], ball: Point, team_ids: Collection[TeamId],
                    max_distance: float, min_margin: float, *, lower_region: bool = False) -> tuple[Contact | None, str | None]:
    ball_x, ball_y = ball
    nearby = []
    for item in tracks:
        class_id, track_id = item['class_id'], item['track_id']
        if (type(class_id) is not int or class_id not in (0, 1) or type(track_id) is not int
                or track_id <= 0 or item.get('observed') is not True):
            continue
        bounds = track_box(item)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        height = y2 - y1
        relative = (0., 0.)
        if lower_region:
            horizontal_scale = max(height, 2 * (x2 - x1))
            dx = max(x1 - ball_x, ball_x - x2, 0.) / horizontal_scale
            dy = max(y2 - .20 * height - ball_y, ball_y - y2, 0.) / height
            distance = math.hypot(dx, dy)
            relative = ((ball_x - (x1 + x2) / 2) / horizontal_scale, (ball_y - y2) / height)
        else:
            distance = math.hypot(ball_x - (x1 + x2) / 2, ball_y - y2) / height
        if distance <= max_distance:
            team = item.get('team')
            team = team if type(team) in (int, str) and team in team_ids else None
            nearby.append(Contact((segment, class_id, track_id), team, distance, relative))
    if not nearby:
        return None, 'no_nearby_player'
    nearby.sort(key=lambda item: item.distance)
    if len(nearby) > 1:
        margin = nearby[1].distance - nearby[0].distance
        threshold = min_margin
        if margin == 0 or (margin < threshold and not math.isclose(margin, threshold, abs_tol=1e-12)):
            return None, 'ambiguous_owner'
    return nearby[0], None
