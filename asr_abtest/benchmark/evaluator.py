from typing import List, Dict

class WERCalculator:
    @staticmethod
    def calculate(reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate between two texts"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Create matrix
        d = [[0 for _ in range(len(ref_words) + 1)] 
             for _ in range(len(hyp_words) + 1)]
        
        # Initialize first row and column
        for i in range(len(hyp_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j
            
        # Fill matrix
        for i in range(1, len(hyp_words) + 1):
            for j in range(1, len(ref_words) + 1):
                if hyp_words[i-1] == ref_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
                    
        return d[len(hyp_words)][len(ref_words)] / len(ref_words)
    
    @staticmethod
    def analyze_errors(reference: str, hypothesis: str) -> Dict:
        """Analyze types of errors in the transcription"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Initialize counters
        substitutions = 0
        deletions = 0
        insertions = 0
        
        # Create alignment matrix
        m = len(hyp_words) + 1
        n = len(ref_words) + 1
        d = [[0] * n for _ in range(m)]
        
        # Initialize first row and column
        for i in range(m):
            d[i][0] = i
        for j in range(n):
            d[0][j] = j
        
        # Fill the matrix and track operations
        for i in range(1, m):
            for j in range(1, n):
                if hyp_words[i-1] == ref_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    deletion = d[i-1][j] + 1
                    insertion = d[i][j-1] + 1
                    d[i][j] = min(substitution, deletion, insertion)
                    
                    # Count error types
                    if d[i][j] == substitution:
                        substitutions += 1
                    elif d[i][j] == deletion:
                        deletions += 1
                    else:
                        insertions += 1
        
        total_errors = substitutions + deletions + insertions
        total_words = len(ref_words)
        
        return {
            "total_errors": total_errors,
            "total_words": total_words,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "error_rate": total_errors / total_words if total_words > 0 else 1.0,
            "accuracy": 1 - (total_errors / total_words if total_words > 0 else 1.0)
        } 