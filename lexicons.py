LOWERCASE_ROMAN = tuple("abcdefghijklmnopqrstuvwxyz")
UPPERCASE_ROMAN = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMERALS = tuple("0123456789")
LOWERCASE_GREEK = (r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta", r"\eta", r"\theta",
                   r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau",
                   r"\phi", r"\chi", r"\psi", r"\omega")
UPPERCASE_GREEK = (r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi", r"\Phi", r"\Psi", r"\Omega")
SYMBOLS = LOWERCASE_ROMAN + UPPERCASE_ROMAN + NUMERALS + LOWERCASE_GREEK + UPPERCASE_GREEK


SPECIAL_CHARACTERS = ("(", ")", "+", "-", "=")
TEX_CONTROL_SEQUENCES = (r"\cdot", r"\left(", r"\right)", r"\dfrac{", "}{", r"\sqrt{", "}", "^{", "_{")
CONTROL_CHARACTERS = ("EOL", )


TOKENS = SYMBOLS + SPECIAL_CHARACTERS + TEX_CONTROL_SEQUENCES


ALL_TOKENS: tuple[str] = TOKENS + CONTROL_CHARACTERS
