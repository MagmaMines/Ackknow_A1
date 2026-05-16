from .components.modules import Confic, Guider, IQA, Searcher, Store, Unip
from .config.settings import AckknowSettings
from .datasets.catalog import load_default_catalog
from .types import AckResponse, QueryContext


class AckknowA1:
    def __init__(self, settings: AckknowSettings | None = None) -> None:
        self.settings = settings or AckknowSettings()
        self.store = Store(memory={})
        self.searcher = Searcher()
        self.iqa = IQA()
        self.confic = Confic()
        self.guider = Guider()
        self.unip = Unip()
        self.datasets = load_default_catalog()

    def run(self, prompt: str, ctx: QueryContext) -> AckResponse:
        history = self.store.recall(ctx)
        evidence = self.searcher.retrieve(prompt)
        reasoning = self.iqa.reason(prompt, evidence)
        pattern_report = self.confic.analyze_patterns(prompt)
        response = self.unip.synthesize(prompt, history, reasoning, pattern_report, evidence)
        response.safeguards = self.guider.safeguards()
        self.store.persist(ctx, prompt)
        return response
