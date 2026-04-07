import argparse
import base64
import html as html_module
import os
import random as pyrandom
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from sage.all import *


@dataclass
class Scene:
    graphics: object
    renderer: str
    title: str
    report_lines: list[str]
    save_kwargs: dict = field(default_factory=dict)


@dataclass
class RefinementStep2D:
    step_index: int
    facet_label: str
    parent_edge: tuple
    P: object
    Q: object
    child_edges: list[tuple]


@dataclass
class RefinementStep3D:
    step_index: int
    facet_label: str
    parent_face: tuple
    P: object
    Q: object
    child_faces: list[tuple]


POINT_MARKER_SIZE = 15
ORIGIN_MARKER_SIZE = 10

SEVEN_SEGMENT_STROKES = {
    "0": ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
    "1": ("upper_right", "lower_right"),
    "2": ("top", "upper_right", "middle", "lower_left", "bottom"),
    "3": ("top", "upper_right", "middle", "lower_right", "bottom"),
    "4": ("upper_left", "upper_right", "middle", "lower_right"),
    "5": ("top", "upper_left", "middle", "lower_right", "bottom"),
    "6": ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
    "7": ("top", "upper_right", "lower_right"),
    "8": (
        "top",
        "upper_left",
        "upper_right",
        "middle",
        "lower_left",
        "lower_right",
        "bottom",
    ),
    "9": ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
}

SEGMENT_COORDS = {
    "top": ((0.0, 1.0), (1.0, 1.0)),
    "upper_left": ((0.0, 1.0), (0.0, 0.5)),
    "upper_right": ((1.0, 1.0), (1.0, 0.5)),
    "middle": ((0.0, 0.5), (1.0, 0.5)),
    "lower_left": ((0.0, 0.5), (0.0, 0.0)),
    "lower_right": ((1.0, 0.5), (1.0, 0.0)),
    "bottom": ((0.0, 0.0), (1.0, 0.0)),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the radial-projection examples in one CLI. "
            "Default mode modifies all tetrahedron faces; "
            "--2d switches to the triangle-in-circle analogue; "
            "--1-face only changes the display so one modified facet "
            "is highlighted while the full object is still built."
        )
    )
    parser.add_argument(
        "--2d",
        dest="two_d",
        action="store_true",
        help="Use the 2D triangle-in-circle model instead of the default 3D tetrahedron.",
    )
    parser.add_argument(
        "--1-face",
        dest="one_face",
        action="store_true",
        help=(
            "Display only one modified facet while keeping the underlying "
            "construction identical to the default mode."
        ),
    )
    parser.add_argument(
        "--1-dface",
        dest="one_display_face",
        action="store_true",
        help=(
            "Alias for --1-face."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set the random seed for reproducible random choices.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_iterations",
        type=int,
        metavar="X",
        help=(
            "Start from the raw simplex and perform X random boundary refinements, "
            "always choosing from the full current boundary."
        ),
    )
    parser.add_argument(
        "--output",
        help="Write the output HTML to this path. Defaults to output-<timestamp>.html.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write the file without calling xdg-open.",
    )
    parser.add_argument(
        "--degenerate",
        action="store_true",
        help="In all-faces mode, place one chosen facet point on an edge.",
    )
    parser.add_argument(
        "--near-edge",
        action="store_true",
        help="In 3D --1-face mode, choose P near an edge to bias toward nonconvex examples.",
    )
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]

    args = parser.parse_args(argv)

    if args.continue_iterations is not None and args.continue_iterations < 0:
        parser.error("--continue requires a nonnegative integer.")
    if args.degenerate and args.two_d:
        parser.error("--degenerate only applies to the 3D construction.")
    if args.degenerate and args.continue_iterations is not None:
        parser.error("--degenerate cannot be combined with --continue.")
    if args.near_edge:
        parser.error(
            "--near-edge is not supported because --1-face is display-only "
            "and no longer changes the construction."
        )

    return args


def configure_randomness(seed):
    if seed is None:
        pyrandom.seed()
        return

    pyrandom.seed(seed)
    set_random_seed(seed)


def default_output_path(requested_output):
    if requested_output:
        output_path = Path(requested_output).expanduser()
        if output_path.suffix.lower() != ".html":
            output_path = output_path.with_name(output_path.name + ".html")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"output-{timestamp}.html")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def weighted_sum(weights, points):
    return sum((weight * point for weight, point in zip(weights, points)), points[0] * 0)


def random_point_in_triangle(A, B, C, margin=0.0):
    weights = [pyrandom.random() for _ in range(3)]
    total = sum(weights)
    weights = [weight / total for weight in weights]

    if margin:
        factor = 1 - 3 * margin
        weights = [margin + factor * weight for weight in weights]

    return weighted_sum(weights, (A, B, C))


def random_point_on_edge(A, B, t_min=0.2, t_max=0.8):
    t = pyrandom.uniform(t_min, t_max)
    return (1 - t) * A + t * B


def normal_of_plane(A, B, C):
    n = (B - A).cross_product(C - A)
    return n / n.norm()


def orthogonal_projection_to_plane(X, A, n):
    return X - (X - A).dot_product(n) * n


def orthogonal_projection_to_line(X, A, B):
    direction = B - A
    t = (X - A).dot_product(direction) / direction.dot_product(direction)
    return A + t * direction


def barycentric_coordinates_in_triangle(P, A, B, C):
    v0 = B - A
    v1 = C - A
    v2 = P - A

    d00 = v0.dot_product(v0)
    d01 = v0.dot_product(v1)
    d11 = v1.dot_product(v1)
    d20 = v2.dot_product(v0)
    d21 = v2.dot_product(v1)

    denom = d00 * d11 - d01 * d01
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1 - beta - gamma
    return alpha, beta, gamma


def point_in_triangle(P, A, B, C, eps=1e-9):
    alpha, beta, gamma = barycentric_coordinates_in_triangle(P, A, B, C)
    return all(float(N(value)) >= -eps for value in (alpha, beta, gamma))


def segment_parameter(P, A, B):
    direction = B - A
    return (P - A).dot_product(direction) / direction.dot_product(direction)


def point_on_segment(P, A, B, eps=1e-9):
    t = float(N(segment_parameter(P, A, B)))
    return -eps <= t <= 1 + eps


def supporting_triangle(tri, all_vertices, interior_point, eps=1e-9):
    A, B, C = tri
    n = (B - A).cross_product(C - A)

    if n.norm() < eps:
        return False

    if n.dot_product(interior_point - A) > 0:
        n = -n

    for V in all_vertices:
        if n.dot_product(V - A) > eps:
            return False

    return True


def is_convex_boundary(boundary_triangles, all_vertices, eps=1e-9):
    center = sum(all_vertices) / len(all_vertices)
    bad_faces = []

    for index, tri in enumerate(boundary_triangles):
        if not supporting_triangle(tri, all_vertices, center, eps=eps):
            bad_faces.append(index)

    return len(bad_faces) == 0, bad_faces


def supporting_segment(seg, all_vertices, interior_point, eps=1e-9):
    A, B = seg
    direction = B - A
    n = vector((direction[1], -direction[0]))

    if n.norm() < eps:
        return False

    if n.dot_product(interior_point - A) > 0:
        n = -n

    for V in all_vertices:
        if n.dot_product(V - A) > eps:
            return False

    return True


def is_convex_polygon(boundary_segments, all_vertices, eps=1e-9):
    center = sum(all_vertices) / len(all_vertices)
    bad_edges = []

    for index, seg in enumerate(boundary_segments):
        if not supporting_segment(seg, all_vertices, center, eps=eps):
            bad_edges.append(index)

    return len(bad_edges) == 0, bad_edges


def regular_triangle():
    half = SR(1) / 2
    return [
        vector((0, 1)),
        vector((-sqrt(3) / 2, -half)),
        vector((sqrt(3) / 2, -half)),
    ]


def triangle_edges(vertices):
    A, B, C = vertices
    return [
        ("E1", (A, B), C),
        ("E2", (B, C), A),
        ("E3", (C, A), B),
    ]


def regular_tetrahedron():
    V1 = vector((1, 1, 1)) / sqrt(3)
    V2 = vector((1, -1, -1)) / sqrt(3)
    V3 = vector((-1, 1, -1)) / sqrt(3)
    V4 = vector((-1, -1, 1)) / sqrt(3)
    return [V1, V2, V3, V4]


def tetrahedron_edges(vertices):
    V1, V2, V3, V4 = vertices
    return [
        (V1, V2),
        (V1, V3),
        (V1, V4),
        (V2, V3),
        (V2, V4),
        (V3, V4),
    ]


def tetrahedron_facets(vertices):
    V1, V2, V3, V4 = vertices
    return [
        ("F1", (V1, V2, V3), V4),
        ("F2", (V1, V2, V4), V3),
        ("F3", (V1, V3, V4), V2),
        ("F4", (V2, V3, V4), V1),
    ]


def add_tetrahedron_facets(graphics, vertices, color="#c4c4c4", opacity=0.10):
    for _, (A, B, C), _ in tetrahedron_facets(vertices):
        graphics += polygon3d([A, B, C], color=color, opacity=opacity)


def add_tetrahedron_wireframe(graphics, vertices, labels=None):
    for X, Y in tetrahedron_edges(vertices):
        graphics += line3d([X, Y], color="black", thickness=2)

    for index, vertex in enumerate(vertices):
        graphics += point3d(vertex, size=POINT_MARKER_SIZE, color="black")
        if labels is not None:
            graphics += text3d(
                labels[index],
                vertex + vector((0.03, 0.03, 0.03)),
                color="black",
            )


def add_triangle_wireframe(graphics, vertices, labels=None):
    A, B, C = vertices
    for X, Y in [(A, B), (B, C), (C, A)]:
        graphics += line([X, Y], color="black", thickness=2)

    for index, vertex in enumerate(vertices):
        graphics += point(vertex, size=POINT_MARKER_SIZE, color="black")
        if labels is not None:
            graphics += text(labels[index], vertex + vector((0.04, 0.04)), color="black")


def format_coords(coords, digits=6):
    return "(" + ", ".join(f"{float(N(value)):.{digits}f}" for value in coords) + ")"


def draw_digit_segments_2d(graphics, label, anchor, scale=0.06, color="black", thickness=2):
    x_cursor = 0.0
    for char in str(label):
        strokes = SEVEN_SEGMENT_STROKES.get(char)
        if strokes is None:
            x_cursor += 1.4
            continue

        for segment_name in strokes:
            (x1, y1), (x2, y2) = SEGMENT_COORDS[segment_name]
            start = anchor + vector(((x_cursor + x1) * scale, y1 * scale))
            end = anchor + vector(((x_cursor + x2) * scale, y2 * scale))
            graphics += line([start, end], color=color, thickness=thickness)
        x_cursor += 1.4


def digit_box_dimensions(label, scale):
    text = str(label)
    width_units = 1.0 + 1.4 * max(len(text) - 1, 0)
    width = scale * (width_units + 0.4)
    height = scale * 1.3
    return width, height


def tangent_basis_3d(point):
    normal = point / point.norm()
    reference = vector((0, 0, 1))
    if abs(float(N(normal.dot_product(reference)))) > 0.9:
        reference = vector((0, 1, 0))

    u = reference.cross_product(normal)
    u = u / u.norm()
    v = normal.cross_product(u)
    v = v / v.norm()
    return normal, u, v


def draw_digit_segments_3d(graphics, label, anchor, u, v, scale=0.07, color="black", thickness=3):
    x_cursor = 0.0
    for char in str(label):
        strokes = SEVEN_SEGMENT_STROKES.get(char)
        if strokes is None:
            x_cursor += 1.4
            continue

        for segment_name in strokes:
            (x1, y1), (x2, y2) = SEGMENT_COORDS[segment_name]
            start = anchor + ((x_cursor + x1) * scale) * u + (y1 * scale) * v
            end = anchor + ((x_cursor + x2) * scale) * u + (y2 * scale) * v
            graphics += line3d([start, end], color=color, thickness=thickness)
        x_cursor += 1.4


def simulate_2d_batch():
    vertices = regular_triangle()
    active_edges = []
    all_vertices = list(vertices)
    steps = []

    for step_index, (label, (A, B), _) in enumerate(triangle_edges(vertices), start=1):
        P = random_point_on_edge(A, B)
        Q = P / P.norm()
        child_edges = [(A, Q), (Q, B)]
        active_edges.extend(child_edges)
        all_vertices.append(Q)
        steps.append(
            RefinementStep2D(
                step_index=step_index,
                facet_label=label,
                parent_edge=(A, B),
                P=P,
                Q=Q,
                child_edges=child_edges,
            )
        )

    return vertices, active_edges, all_vertices, steps, 0


def simulate_2d_continue(iterations):
    vertices = regular_triangle()
    active_pool = [
        {"label": label, "edge": edge}
        for label, edge, _ in triangle_edges(vertices)
    ]
    all_vertices = list(vertices)
    steps = []
    next_label = len(active_pool) + 1

    for step_index in range(1, iterations + 1):
        chosen_index = pyrandom.randrange(len(active_pool))
        chosen = active_pool.pop(chosen_index)
        A, B = chosen["edge"]
        P = random_point_on_edge(A, B)
        Q = P / P.norm()
        child_edges = [(A, Q), (Q, B)]
        active_pool.extend(
            [
                {"label": f"E{next_label}", "edge": child_edges[0]},
                {"label": f"E{next_label + 1}", "edge": child_edges[1]},
            ]
        )
        next_label += 2
        all_vertices.append(Q)
        steps.append(
            RefinementStep2D(
                step_index=step_index,
                facet_label=chosen["label"],
                parent_edge=(A, B),
                P=P,
                Q=Q,
                child_edges=child_edges,
            )
        )

    return vertices, [item["edge"] for item in active_pool], all_vertices, steps, len(steps) - 1


def draw_2d_focus_overlay(graphics, step):
    A, B = step.parent_edge
    graphics += line([A, B], color="orange", thickness=2, linestyle="--")
    for edge in step.child_edges:
        graphics += line(list(edge), color="orange", thickness=4)

    graphics += point(step.P, size=POINT_MARKER_SIZE, color="orange")
    graphics += point(step.Q, size=POINT_MARKER_SIZE, color="orange")
    graphics += text("P", step.P + vector((0.04, 0.04)), color="orange")
    graphics += text("Q", step.Q + vector((0.04, 0.04)), color="orange")
    graphics += line([(0, 0), step.Q], color="orange", linestyle="--", thickness=1)


def add_2d_continue_labels(graphics, steps, focus_step=None):
    for step in steps:
        label_color = (
            "orange"
            if focus_step is not None and step.step_index == focus_step.step_index
            else "#9a1f14"
        )
        scale = 0.10
        width, height = digit_box_dimensions(step.step_index, scale)
        anchor = 0.68 * step.Q + vector((0.015, -0.04))
        box = [
            anchor + vector((-0.02, -0.02)),
            anchor + vector((width + 0.02, -0.02)),
            anchor + vector((width + 0.02, height + 0.02)),
            anchor + vector((-0.02, height + 0.02)),
        ]
        box_center = anchor + vector((width / 2, height / 2))
        graphics += line([step.Q, box_center], color=label_color, thickness=2, linestyle="--")
        graphics += polygon(box, color="white", alpha=0.92)
        graphics += line(box + [box[0]], color=label_color, thickness=2)
        draw_digit_segments_2d(
            graphics,
            step.step_index,
            anchor,
            scale=scale,
            color=label_color,
            thickness=4,
        )


def build_2d_refinement_scene(continue_iterations=None, display_one_edge=False):
    if continue_iterations is None:
        vertices, active_edges, all_vertices, steps, focus_index = simulate_2d_batch()
        mode_label = "2d all-edges"
        title = "2D All-Edge Augmentation"
    else:
        vertices, active_edges, all_vertices, steps, focus_index = simulate_2d_continue(
            continue_iterations
        )
        mode_label = "2d continue"
        title = "2D Iterative Refinement"

    focus_step = None
    if display_one_edge and steps:
        focus_step = steps[focus_index]

    convex, bad_edges = is_convex_polygon(active_edges, all_vertices)

    G = Graphics()
    G += circle((0, 0), 1, thickness=2, color="steelblue")
    G += polygon(vertices, color="lightgray", alpha=0.10)
    G += point((0, 0), size=ORIGIN_MARKER_SIZE, color="black")
    G += text("O", vector((0.04, 0.04)), color="black")
    add_triangle_wireframe(G, vertices, labels=["A", "B", "C"])

    boundary_color = "#8f5a2a" if focus_step is None else "#7a7a7a"
    boundary_thickness = 3 if focus_step is None else 2
    for edge in active_edges:
        G += line(list(edge), color=boundary_color, thickness=boundary_thickness)

    vertex_color = "#355caa" if focus_step is None else "#7d90c7"
    for vertex in all_vertices[len(vertices):]:
        G += point(vertex, size=POINT_MARKER_SIZE, color=vertex_color)

    if focus_step is not None:
        draw_2d_focus_overlay(G, focus_step)

    if continue_iterations is not None:
        add_2d_continue_labels(G, steps, focus_step=focus_step)

    report_lines = [
        f"mode: {mode_label}",
        f"convex: {convex}",
        f"current boundary segments: {len(active_edges)}",
    ]
    if continue_iterations is not None:
        report_lines.append(f"iterations: {continue_iterations}")
    else:
        report_lines.append(f"initial refinements: {len(steps)}")
    if focus_step is not None:
        report_lines.append(
            f"displayed refinement: step {focus_step.step_index} on {focus_step.facet_label}"
        )
    if not convex:
        report_lines.append(f"non-supporting boundary segments: {bad_edges}")

    return Scene(
        graphics=G,
        renderer="embedded-png-html",
        title=title,
        report_lines=report_lines,
        save_kwargs={"axes": False, "aspect_ratio": 1},
    )


def simulate_3d_batch(degenerate=False):
    vertices = regular_tetrahedron()
    active_faces = []
    all_vertices = list(vertices)
    steps = []
    facets = tetrahedron_facets(vertices)
    degenerate_index = pyrandom.randrange(len(facets)) if degenerate else None
    focus_index = degenerate_index if degenerate else 0

    for step_index, (label, (A, B, C), _) in enumerate(facets, start=1):
        if degenerate and step_index - 1 == degenerate_index:
            chosen_edge = pyrandom.choice([(A, B), (A, C), (B, C)])
            P = random_point_on_edge(*chosen_edge)
        else:
            P = random_point_in_triangle(A, B, C, margin=0.08)

        Q = P / P.norm()
        child_faces = [(A, B, Q), (B, C, Q), (C, A, Q)]
        active_faces.extend(child_faces)
        all_vertices.append(Q)
        steps.append(
            RefinementStep3D(
                step_index=step_index,
                facet_label=label,
                parent_face=(A, B, C),
                P=P,
                Q=Q,
                child_faces=child_faces,
            )
        )

    return vertices, active_faces, all_vertices, steps, focus_index, degenerate_index


def simulate_3d_continue(iterations):
    vertices = regular_tetrahedron()
    active_pool = [
        {"label": label, "face": face}
        for label, face, _ in tetrahedron_facets(vertices)
    ]
    all_vertices = list(vertices)
    steps = []
    next_label = len(active_pool) + 1

    for step_index in range(1, iterations + 1):
        chosen_index = pyrandom.randrange(len(active_pool))
        chosen = active_pool.pop(chosen_index)
        A, B, C = chosen["face"]
        P = random_point_in_triangle(A, B, C, margin=0.08)
        Q = P / P.norm()
        child_faces = [(A, B, Q), (B, C, Q), (C, A, Q)]
        active_pool.extend(
            [
                {"label": f"F{next_label}", "face": child_faces[0]},
                {"label": f"F{next_label + 1}", "face": child_faces[1]},
                {"label": f"F{next_label + 2}", "face": child_faces[2]},
            ]
        )
        next_label += 3
        all_vertices.append(Q)
        steps.append(
            RefinementStep3D(
                step_index=step_index,
                facet_label=chosen["label"],
                parent_face=(A, B, C),
                P=P,
                Q=Q,
                child_faces=child_faces,
            )
        )

    return vertices, [item["face"] for item in active_pool], all_vertices, steps, len(steps) - 1


def draw_3d_focus_overlay(graphics, step):
    A, B, C = step.parent_face
    graphics += polygon3d([A, B, C], color="orange", opacity=0.10)
    for face in step.child_faces:
        graphics += polygon3d(list(face), color="orange", opacity=0.28)
        for X, Y in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            graphics += line3d([X, Y], color="orange", thickness=3)

    graphics += point3d(step.P, size=POINT_MARKER_SIZE, color="orange")
    graphics += point3d(step.Q, size=POINT_MARKER_SIZE, color="orange")
    graphics += text3d("P", step.P + vector((0.03, 0.03, 0.03)), color="orange")
    graphics += text3d("Q", step.Q + vector((0.03, 0.03, 0.03)), color="orange")
    graphics += line3d([(0, 0, 0), step.Q], color="orange", thickness=1, linestyle="--")


def add_3d_continue_labels(graphics, steps, focus_step=None):
    for step in steps:
        label_color = (
            "orange"
            if focus_step is not None and step.step_index == focus_step.step_index
            else "#9a1f14"
        )
        normal, u, v = tangent_basis_3d(step.Q)
        scale = 0.20
        width, height = digit_box_dimensions(step.step_index, scale)
        center = 1.32 * step.Q + 0.18 * u + 0.18 * v
        lower_left = center - (width / 2) * u - (height / 2) * v
        panel = [
            lower_left + (-0.03) * u + (-0.03) * v,
            lower_left + (width + 0.03) * u + (-0.03) * v,
            lower_left + (width + 0.03) * u + (height + 0.03) * v,
            lower_left + (-0.03) * u + (height + 0.03) * v,
        ]
        graphics += line3d([step.Q, center], color=label_color, thickness=5)
        graphics += polygon3d(panel, color="white", opacity=0.92)
        for X, Y in zip(panel, panel[1:] + panel[:1]):
            graphics += line3d([X, Y], color=label_color, thickness=4)
        draw_digit_segments_3d(
            graphics,
            step.step_index,
            lower_left,
            u,
            v,
            scale=scale,
            color=label_color,
            thickness=10,
        )


def build_3d_refinement_scene(
    continue_iterations=None,
    display_one_face=False,
    degenerate=False,
):
    if continue_iterations is None:
        (
            vertices,
            active_faces,
            all_vertices,
            steps,
            focus_index,
            degenerate_index,
        ) = simulate_3d_batch(degenerate=degenerate)
        mode_label = "3d all-faces"
        title = "3D All-Face Augmentation"
    else:
        vertices, active_faces, all_vertices, steps, focus_index = simulate_3d_continue(
            continue_iterations
        )
        degenerate_index = None
        mode_label = "3d continue"
        title = "3D Iterative Refinement"

    focus_step = None
    if display_one_face and steps:
        focus_step = steps[focus_index]

    convex, bad_faces = is_convex_boundary(active_faces, all_vertices)

    G = Graphics()
    G += sphere((0, 0, 0), 1, color="lightblue", opacity=0.16)
    add_tetrahedron_facets(G, vertices, color="#d9d9d9", opacity=0.06)
    add_tetrahedron_wireframe(G, vertices)

    face_color = "#c57a33" if focus_step is None else "#b9b9b9"
    edge_color = "#734924" if focus_step is None else "#7a7a7a"
    face_opacity = 0.18 if focus_step is None else 0.10
    for face in active_faces:
        G += polygon3d(list(face), color=face_color, opacity=face_opacity)
        for X, Y in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            G += line3d([X, Y], color=edge_color, thickness=2)

    vertex_color = "#355caa" if focus_step is None else "#7d90c7"
    for vertex in all_vertices[len(vertices):]:
        G += point3d(vertex, size=POINT_MARKER_SIZE, color=vertex_color)

    if focus_step is not None:
        draw_3d_focus_overlay(G, focus_step)

    if continue_iterations is not None:
        add_3d_continue_labels(G, steps, focus_step=focus_step)

    report_lines = [
        f"mode: {mode_label}",
        f"convex: {convex}",
        f"current boundary triangles: {len(active_faces)}",
    ]
    if continue_iterations is not None:
        report_lines.append(f"iterations: {continue_iterations}")
    else:
        report_lines.append(f"initial refinements: {len(steps)}")
    if focus_step is not None:
        report_lines.append(
            f"displayed refinement: step {focus_step.step_index} on {focus_step.facet_label}"
        )
    if not convex:
        report_lines.append(f"non-supporting boundary triangles: {bad_faces}")
    if degenerate_index is not None:
        report_lines.append(f"degenerate facet: F{degenerate_index + 1}")

    return Scene(
        graphics=G,
        renderer="sage-html",
        title=title,
        report_lines=report_lines,
        save_kwargs={"online": True},
    )


def build_2d_all_edges_scene(display_one_edge=False):
    vertices = regular_triangle()
    edges = triangle_edges(vertices)
    edge_colors = ["red", "green", "orange"]
    display_edge_index = 0

    G = Graphics()
    G += circle((0, 0), 1, thickness=2, color="steelblue")
    G += polygon(vertices, color="lightgray", alpha=0.10)
    G += point((0, 0), size=ORIGIN_MARKER_SIZE, color="black")
    G += text("O", vector((0.04, 0.04)), color="black")
    add_triangle_wireframe(G, vertices, labels=["A", "B", "C"])

    projected_points = []
    boundary_segments = []

    for index, (name, (A, B), _) in enumerate(edges):
        color = edge_colors[index % len(edge_colors)]
        P = random_point_on_edge(A, B)
        Q = P / P.norm()
        projected_points.append(Q)

        if not display_one_edge or index == display_edge_index:
            G += line([A, B], color=color, thickness=3)
            G += point(P, size=POINT_MARKER_SIZE, color=color)
            G += point(Q, size=POINT_MARKER_SIZE, color=color)
            G += text(f"P{index + 1}", P + vector((0.04, 0.04)), color=color)
            G += text(f"Q{index + 1}", Q + vector((0.04, 0.04)), color=color)
            G += line([(0, 0), Q], color=color, linestyle="--", thickness=1)
            G += line([A, Q], color=color, thickness=3)
            G += line([Q, B], color=color, thickness=3)

        boundary_segments.extend([(A, Q), (Q, B)])

    all_vertices = vertices + projected_points
    convex, bad_edges = is_convex_polygon(boundary_segments, all_vertices)

    report_lines = [
        "mode: 2d all-edges",
        f"convex: {convex}",
    ]
    if display_one_edge:
        report_lines.append(f"displayed edge only: {edges[display_edge_index][0]}")
    if not convex:
        report_lines.append(f"non-supporting boundary segments: {bad_edges}")

    return Scene(
        graphics=G,
        renderer="embedded-png-html",
        title="2D All-Edge Augmentation",
        report_lines=report_lines,
        save_kwargs={"axes": False, "aspect_ratio": 1},
    )


def build_2d_one_edge_scene(display_one_edge=False):
    A, B, C = regular_triangle()
    P = random_point_on_edge(A, B)
    Q = P / P.norm()
    R = orthogonal_projection_to_line(Q, A, B)
    t = float(N(segment_parameter(R, A, B)))
    r_in_edge = point_on_segment(R, A, B)

    G = Graphics()
    G += circle((0, 0), 1, thickness=2, color="steelblue")
    G += polygon([A, B, C], color="lightgray", alpha=0.10)
    G += point((0, 0), size=ORIGIN_MARKER_SIZE, color="black")
    G += text("O", vector((0.04, 0.04)), color="black")
    add_triangle_wireframe(G, [A, B, C], labels=["A", "B", "C"])

    G += line([A, B], color="orange", thickness=3)
    G += point(P, size=POINT_MARKER_SIZE, color="orange")
    G += text("P", P + vector((0.04, 0.04)), color="orange")

    q_color = "green" if r_in_edge else "red"
    G += point(Q, size=POINT_MARKER_SIZE, color=q_color)
    G += text("Q", Q + vector((0.04, 0.04)), color=q_color)

    G += point(R, size=POINT_MARKER_SIZE, color="blue")
    G += text("R", R + vector((0.04, 0.04)), color="blue")

    G += line([(0, 0), Q], color=q_color, linestyle="--", thickness=2)
    G += line([Q, R], color="blue", linestyle="--", thickness=2)
    G += line([A, Q], color=q_color, thickness=3)
    G += line([Q, B], color=q_color, thickness=3)
    G += line([B, C], color="black", thickness=2)
    G += line([C, A], color="black", thickness=2)

    boundary_segments = [(A, Q), (Q, B), (B, C), (C, A)]
    convex, bad_edges = is_convex_polygon(boundary_segments, [A, B, C, Q])

    report_lines = [
        "mode: 2d one-edge",
        f"P = {format_coords(P)}",
        f"Q = {format_coords(Q)}",
        f"R = {format_coords(R)}",
        f"R in segment AB? {r_in_edge}",
        f"segment parameter of R on AB = {t:.6f}",
        f"one-edge augmentation convex? {convex}",
    ]
    if display_one_edge:
        report_lines.append("displayed edge only: E1")
    if not convex:
        report_lines.append(f"non-supporting boundary segments: {bad_edges}")

    return Scene(
        graphics=G,
        renderer="embedded-png-html",
        title="2D One-Edge Augmentation",
        report_lines=report_lines,
        save_kwargs={"axes": False, "aspect_ratio": 1},
    )


def build_3d_all_faces_scene(degenerate=False, display_one_face=False):
    vertices = regular_tetrahedron()
    facets = tetrahedron_facets(vertices)
    degenerate_facet_index = pyrandom.randrange(len(facets)) if degenerate else None
    facet_colors = ["red", "green", "orange", "purple"]
    display_facet_index = degenerate_facet_index if degenerate else 0

    G = Graphics()
    G += sphere((0, 0, 0), 1, color="lightblue", opacity=0.16)
    if display_one_face:
        add_tetrahedron_facets(G, vertices)
    add_tetrahedron_wireframe(G, vertices)

    projected_points = []
    boundary_triangles = []

    for index, (name, (A, B, C), _) in enumerate(facets):
        color = facet_colors[index % len(facet_colors)]
        show_facet = not display_one_face or index == display_facet_index
        if show_facet:
            G += polygon3d([A, B, C], color=color, opacity=0.08)
            G += line3d([A, B], color=color, thickness=3)
            G += line3d([B, C], color=color, thickness=3)
            G += line3d([C, A], color=color, thickness=3)

        if degenerate and index == degenerate_facet_index:
            chosen_edge = pyrandom.choice([(A, B), (A, C), (B, C)])
            P = random_point_on_edge(*chosen_edge)
        else:
            chosen_edge = None
            P = random_point_in_triangle(A, B, C, margin=0.08)

        Q = P / P.norm()
        projected_points.append(Q)

        if show_facet:
            G += point3d(P, size=POINT_MARKER_SIZE, color=color)
            G += point3d(Q, size=POINT_MARKER_SIZE, color=color)
            G += line3d([(0, 0, 0), Q], color=color, thickness=1, linestyle="--")
            G += line3d([A, Q], color=color, thickness=3)
            G += line3d([B, Q], color=color, thickness=3)
            G += line3d([C, Q], color=color, thickness=3)

        triangles = [(A, B, Q), (B, C, Q), (C, A, Q)]
        boundary_triangles.extend(triangles)

        if show_facet:
            for triangle in triangles:
                G += polygon3d(list(triangle), color=color, opacity=0.20)

        if show_facet and chosen_edge is not None:
            G += line3d(list(chosen_edge), color="black", thickness=5)

    all_vertices = vertices + projected_points
    convex, bad_faces = is_convex_boundary(boundary_triangles, all_vertices)

    report_lines = [
        "mode: 3d all-faces",
        f"convex: {convex}",
    ]
    if display_one_face:
        report_lines.append(f"displayed facet only: {facets[display_facet_index][0]}")
    if not convex:
        report_lines.append(f"non-supporting boundary triangles: {bad_faces}")
    if degenerate:
        report_lines.append(f"degenerate facet: {facets[degenerate_facet_index][0]}")

    return Scene(
        graphics=G,
        renderer="sage-html",
        title="3D All-Face Augmentation",
        report_lines=report_lines,
        save_kwargs={"online": True},
    )


def build_3d_one_face_scene(near_edge=False, display_one_face=False):
    A, B, C, D = regular_tetrahedron()
    vertices = [A, B, C, D]

    if near_edge:
        P = 0.48 * A + 0.48 * B + 0.04 * C
    else:
        P = random_point_in_triangle(A, B, C)

    Q = P / P.norm()

    n = normal_of_plane(A, B, C)
    if (D - A).dot_product(n) > 0:
        n = -n

    R = orthogonal_projection_to_plane(Q, A, n)
    alpha, beta, gamma = barycentric_coordinates_in_triangle(R, A, B, C)
    convex = point_in_triangle(R, A, B, C)

    G = Graphics()
    G += sphere((0, 0, 0), 1, color="lightblue", opacity=0.14)
    add_tetrahedron_facets(G, vertices)
    add_tetrahedron_wireframe(G, vertices, labels=["A", "B", "C", "D"])

    G += polygon3d([A, B, C], color="orange", opacity=0.08)

    G += point3d(P, size=POINT_MARKER_SIZE, color="orange")
    G += text3d("P", P + vector((0.03, 0.03, 0.03)), color="orange")

    q_color = "green" if convex else "red"
    G += point3d(Q, size=POINT_MARKER_SIZE, color=q_color)
    G += text3d("Q", Q + vector((0.03, 0.03, 0.03)), color=q_color)

    G += point3d(R, size=POINT_MARKER_SIZE, color="blue")
    G += text3d("R", R + vector((0.03, 0.03, 0.03)), color="blue")

    G += line3d([(0, 0, 0), Q], color=q_color, thickness=2, linestyle="--")
    G += line3d([Q, R], color="blue", thickness=2, linestyle="--")

    for triangle in ([A, B, Q], [B, C, Q], [C, A, Q]):
        G += polygon3d(triangle, color=q_color, opacity=0.22)

    for X, Y in [(A, Q), (B, Q), (C, Q)]:
        G += line3d([X, Y], color=q_color, thickness=3)

    report_lines = [
        "mode: 3d one-face",
        f"P = {format_coords(P)}",
        f"Q = {format_coords(Q)}",
        f"R = {format_coords(R)}",
        f"R in triangle ABC? {convex}",
        f"one-face augmentation convex? {convex}",
        f"barycentric coords of R wrt ABC = {format_coords((alpha, beta, gamma))}",
    ]
    if display_one_face:
        report_lines.append("displayed facet only: F1")

    return Scene(
        graphics=G,
        renderer="sage-html",
        title="3D One-Face Augmentation",
        report_lines=report_lines,
        save_kwargs={"online": True},
    )


def write_embedded_image_html(scene, output_path):
    with TemporaryDirectory() as tmp_dir:
        png_path = Path(tmp_dir) / "scene.png"
        scene.graphics.save(str(png_path), **scene.save_kwargs)
        encoded_image = base64.b64encode(png_path.read_bytes()).decode("ascii")

    report_items = "\n".join(
        f"<li>{html_module.escape(line)}</li>" for line in scene.report_lines
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_module.escape(scene.title)}</title>
  <style>
    body {{
      font-family: sans-serif;
      margin: 2rem;
      background: #f5f6f8;
      color: #20242b;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      background: white;
      padding: 1.5rem;
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      margin-top: 1rem;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html_module.escape(scene.title)}</h1>
    <ul>
      {report_items}
    </ul>
    <img src="data:image/png;base64,{encoded_image}" alt="{html_module.escape(scene.title)}">
  </main>
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")


def save_scene(scene, output_path):
    if scene.renderer == "sage-html":
        scene.graphics.save(str(output_path), **scene.save_kwargs)
        return

    if scene.renderer == "embedded-png-html":
        write_embedded_image_html(scene, output_path)
        return

    raise ValueError(f"Unsupported renderer: {scene.renderer}")


def maybe_open_output(output_path, no_open):
    if no_open:
        return

    if shutil.which("xdg-open") is None:
        print("warning: xdg-open not found; output file was not opened")
        return

    command = f"xdg-open {shlex.quote(str(output_path))} >/dev/null 2>&1 &"
    status = os.system(command)
    if status != 0:
        print(f"warning: xdg-open returned status {status}")


def build_scene(
    two_d=False,
    one_face=False,
    one_display_face=False,
    continue_iterations=None,
    degenerate=False,
    near_edge=False,
):
    display_one_face = one_face or one_display_face

    if continue_iterations is None:
        if two_d:
            return build_2d_all_edges_scene(display_one_edge=display_one_face)
        return build_3d_all_faces_scene(
            degenerate=degenerate,
            display_one_face=display_one_face,
        )

    if two_d:
        return build_2d_refinement_scene(
            continue_iterations=continue_iterations,
            display_one_edge=display_one_face,
        )

    return build_3d_refinement_scene(
        continue_iterations=continue_iterations,
        display_one_face=display_one_face,
    )


def main():
    args = parse_args()
    configure_randomness(args.seed)

    output_path = default_output_path(args.output)
    scene = build_scene(
        two_d=args.two_d,
        one_face=args.one_face,
        one_display_face=args.one_display_face,
        continue_iterations=args.continue_iterations,
        degenerate=args.degenerate,
        near_edge=args.near_edge,
    )

    for line in scene.report_lines:
        print(line)

    save_scene(scene, output_path)
    print(f"wrote {output_path}")
    maybe_open_output(output_path, args.no_open)


if __name__ in {"__main__", "sage.all"}:
    main()
