DB_CONFIG = {'user':'<username>', 'passwd': '<password>', 'host': '<host>'}
RIPSER_LOC = '</path/to/ripser/executable/ripser_tight_representatives>'

mag = {'subjects': [
            'psychology',
            'political science',
            'mathematics',
            'environmental science',
            'computer science',
            'medicine',
            'biology',
            'history',
            'physics',
            'geology',
            'engineering',
            'philosophy',
            'art',
            'sociology',
            'business',
            'economics',
            'chemistry',
            'materials science',
            'geography'
            ], 
            'start': '1900-01-01', 'end': '2021-01-01',
            'levels': list(range(1,6)),
            'default_score_min': 0.6,
            'default_edge_probability_threshold': 0.1}