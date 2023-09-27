from termcolor import colored
import textwrap

class Printer:
    COLORS = {
        "black": 30,
        "grey": 30,  # Actually black but kept for backwards compatibility
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "light_grey": 37,
        "dark_grey": 90,
        "light_red": 91,
        "light_green": 92,
        "light_yellow": 93,
        "light_blue": 94,
        "light_magenta": 95,
        "light_cyan": 96,
        "white": 97,
    }
    text_colors = {
            'info': {
                'text_color': 'magenta',
                'bg_color': None 
            },
            'input': {
                'text_color': 'green',
                'bg_color': None
            },
            'output': {
                'text_color': 'cyan',
                'bg_color': None
            },
            'warning': {
                'text_color': 'yellow',
                'bg_color': None
            },
            'error': {
                'text_color': 'red',
                'bg_color': None
            },
            'critical': {
                'text_color': 'red',
                'bg_color': 'on_white'
            }
        }
    levels = {key:True for key in text_colors.keys()}
    
    def __init__(self) -> None:
        pass

    def set_print_level(self, levels:list):
        levels = [key.lower() for key in levels]
        for key in self.levels:
            if key not in levels:
                self.levels[key]=False
    
    def remove_print_level(self, key:str):
        self.levels[key.lower()]=False
    
    def info(self, message, header='INFO'):
        if self.levels['info'] == False:
            return
        self.__print(header, 'info', str(message))

    def input(self, message, header='INPUT'):
        if self.levels['input'] == False:
            return
        self.__print(header, 'input', str(message))
    
    def output(self, message, header='OUTPUT'):
        if self.levels['output'] == False:
            return
        self.__print(header, 'output', str(message))

    def warning(self, message, header='WARNING'):
        if self.levels['warning'] == False:
            return
        self.__print(header, 'warning', str(message))
    
    def error(self, message, header='ERROR'):
        if self.levels['error'] == False:
            return
        self.__print(header, 'error', str(message))

    def critical(self, message, header='CRITICAL'):
        if self.levels['critical'] == False:
            return
        self.__print(header, 'critical', str(message))

    def __print(self, header, levelname, message):
        indent_width = 20
        text_width = 100
        sub_indent = indent_width*' '
        
        # Wrap the message
        wrapped_lines = textwrap.wrap(message, width=text_width)
        
        # Add level name as title to the first line
        wrapped_lines[0] = header + (indent_width-len(header))*' ' +  wrapped_lines[0]
        
        # Join the wrapped lines
        formatted_string = ('\n'+sub_indent).join(wrapped_lines)
        
        # Print
        print(colored(f"{formatted_string}", 
                      color=self.text_colors[levelname]['text_color'], 
                      on_color=self.text_colors[levelname]['bg_color']))

