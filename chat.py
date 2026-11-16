"""Interactive interface for a trained checkpoint."""
from pathlib import Path
from generator import GeneratorTransformer


def chat():
    """Generate continuations until quit or EOF."""
    model = GeneratorTransformer.load_from_checkpoint(
        Path(__file__).parent / 'results' / 'checkpoint.pt')
    while True:
        try:
            text = input('Вы: ')
        except EOFError:
            return
        if text.strip().lower() == 'quit':
            return
        print('Бот:', model.generate(text, temperature=.8))


if __name__ == '__main__':
    chat()
