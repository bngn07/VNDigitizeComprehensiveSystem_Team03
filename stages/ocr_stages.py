from strategies.base import IBlankDetector

class BlankCheckStage(OCRStage):
    def __init__(self, detector: IBlankDetector):
        super().__init__("BlankCheck")
        self.detector = detector
    
    def handle(self, ctx):
        ctx = self.process(ctx)
        if ctx.is_blank:
            logger.info(f"Short-circuit: {ctx.blank_reason}")
            return ctx                    # KHÔNG gọi self._next
        return self._next.handle(ctx) if self._next else ctx
    
    def process(self, ctx):
        is_blank, score, reason = self.detector.is_blank(ctx.processed_image)
        ctx.is_blank = is_blank
        ctx.blank_score = score
        ctx.blank_reason = reason
        return ctx