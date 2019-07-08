# OLD NAME: corpus

import ast

class Corpus:
    def __init__(self, filename, language):
        with open(filename, 'r', encoding='utf-8') as f:
            dict_syntaxed_string = f.read()
        
        self.stories = ast.literal_eval(dict_syntaxed_string)[language]
        self.class_names = ('animal', 'magic', 'religious', 'realistic', 'orge', 'jokes', 'formula')
        
        def get_number(atu_string):
            try:
                return int(''.join(char for char in atu_string if char.isdigit()))
            except ValueError:
                return -1
        
        def get_atu_range(class_name):
            if class_name == 'animal':
                return (1, 299)
            elif class_name == 'magic':
                return (300, 749)
            elif class_name == 'religious':
                return (750, 849)
            elif class_name == 'realistic':
                return (850, 999)
            elif class_name == 'orge':
                return (1000, 1199)
            elif class_name == 'jokes':
                return (1200, 1999)
            elif class_name == 'formula':
                return (2000, 2399)
            else:
                assert False
        
        def atu_is_in_range(atu_string, class_name):
            minimum, maximum = get_atu_range(class_name)
            return minimum <= get_number(atu_string) <= maximum
        
        def get_stories_of_class(class_name):
            return [story for story in self.stories if atu_is_in_range(story[2], class_name)]
        
        iter_over_class_specific_subsets = (get_stories_of_class(class_name) for class_name in self.class_names)
        
        self.gold_classes = {class_name: stories for class_name, stories in zip(self.class_names, iter_over_class_specific_subsets)}
    
    def __iter__(self):
        return iter(self.stories)
    
