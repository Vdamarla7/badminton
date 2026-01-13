"""
Mock version of poselet classifier for testing when sklearn is not available.
"""

def classify_triplet(point1, point2, point3):
    """
    Mock classification function that returns a default poselet classification.
    
    Args:
        point1: Tuple of (x, y) for the first point
        point2: Tuple of (x, y) for the second point  
        point3: Tuple of (x, y) for the third point
        
    Returns:
        Default poselet classification string
    """
    return "P_0_30"  # Default mock classification