from src import LexiconABSA

text = "The food was great but the waiter was rude."
analyzer = LexiconABSA()
results = analyzer.analyze(text)

for r in results:
    print(f"{r.aspect}: {r.sentiment} ({r.confidence:.2f})")
