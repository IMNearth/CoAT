from colorama import init, Fore, Style


def print_with_color(text, color):
    colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
        'reset': Fore.RESET
    }
    
    color_code = colors.get(color.lower(), Fore.RESET)
    print(color_code + text)

