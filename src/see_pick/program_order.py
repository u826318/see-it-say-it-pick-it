"""Shared deterministic execution order and scene views after completed moves."""

from dataclasses import replace

from .domain import ResolvedProgram, ResolvedStep, SceneGraph, TaskProgram


def ordered_step_indices(program: TaskProgram, resolution: ResolvedProgram) -> tuple[int, ...]:
    """Topologically order original indices; never arbitrarily break a cycle."""
    predecessors = {index: set() for index in range(len(resolution.steps))}
    for link in program.conjunctions:
        if link.kind == "then":
            for before, first in enumerate(resolution.steps):
                for after, second in enumerate(resolution.steps):
                    if first.clause_index <= link.after_clause < second.clause_index:
                        predecessors[after].add(before)
    for edge in resolution.dependencies:
        if edge.before not in predecessors or edge.after not in predecessors or edge.before == edge.after:
            raise ValueError("malformed program dependency")
        predecessors[edge.after].add(edge.before)
    ordered = []
    while predecessors:
        ready = [index for index, required in predecessors.items() if not required]
        if not ready:
            raise ValueError("source order conflicts with physical dependencies")
        index = min(ready, key=lambda i: (resolution.steps[i].clause_index,
                                        resolution.steps[i].object_id, i))
        ordered.append(index)
        del predecessors[index]
        for required in predecessors.values():
            required.discard(index)
    return tuple(ordered)


def ordered_program_steps(program: TaskProgram, resolution: ResolvedProgram) -> tuple[ResolvedStep, ...]:
    return tuple(resolution.steps[index] for index in validated_step_order(program, resolution))


def validated_step_order(program: TaskProgram, resolution: ResolvedProgram,
                         override: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Accept only the canonical order derived from source and dependencies."""
    canonical = ordered_step_indices(program, resolution)
    if override is not None and (not isinstance(override, tuple) or override != canonical):
        raise ValueError("execution order override must equal the canonical program order")
    return canonical


def remaining_scene(graph: SceneGraph, moved: set[str]) -> SceneGraph:
    """Discard moved objects and their incident stored relations and hints."""
    return replace(graph, objects=tuple(
        replace(obj, relation_hints=tuple((predicate, other) for predicate, other in obj.relation_hints
                                         if other not in moved))
        for obj in graph.objects if obj.object_id not in moved),
        relations=tuple(relation for relation in graph.relations
                        if relation.subject_id not in moved and relation.object_id not in moved))
