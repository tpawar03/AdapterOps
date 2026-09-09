"""AdapterOps — multi-task LoRA adapter service.

Subpackages map to the build phases in BUILD-PLAN.md:

    train     QLoRA training, one entrypoint, task as an argument   (Phase 0-1)
    eval      golden-set + hard-cases harness, split path as an arg (Phase 0-1, 4)
    router    (ticket, task) routing, baselines, operating curve    (Phase 2)
    judge     distilled evaluator + calibration                     (Phase 3)
    serve     vLLM multi-LoRA launcher and FastAPI request path     (Phase 0-1)
    manifest  load / pin / promote / roll back a system version     (Phase 4)
"""

__version__ = "0.1.0"
