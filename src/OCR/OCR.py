from abc import ABC, abstractmethod

class OCR(ABC):
    @abstractmethod
    def process(self):
        pass