from IPython.display import display, Markdown, Latex


def markdown(text: str):
    display(Markdown(text))


def latex(text: str):
    display(Latex(text))
