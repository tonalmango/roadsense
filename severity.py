"""Road Sense - Road Damage Severity Classification"""


def calculate_severity(damage_type, confidence):
    """
    Calculate severity based on damage type and detection confidence.

    Parameters:
        damage_type: Detected damage class.
        confidence: YOLO confidence score.

    Returns:
        Severity level: Low, Medium, High, or Critical.
    """

    damage_type = damage_type.lower()

    # Potholes are generally more dangerous
    if damage_type == "pothole":

        if confidence >= 0.85:
            return "Critical"
        elif confidence >= 0.60:
            return "High"
        else:
            return "Medium"

    # Cracks
    elif damage_type == "crack":

        if confidence >= 0.85:
            return "High"
        elif confidence >= 0.60:
            return "Medium"
        else:
            return "Low"

    # Manholes
    elif damage_type == "manhole":

        if confidence >= 0.85:
            return "High"
        elif confidence >= 0.60:
            return "Medium"
        else:
            return "Low"

    # Unknown damage type
    else:

        if confidence >= 0.85:
            return "High"
        elif confidence >= 0.60:
            return "Medium"
        else:
            return "Low"


# --------------------------------------------------
# TEST SEVERITY MODULE
# --------------------------------------------------

if __name__ == "__main__":

    print("Road Sense Severity Module")
    print("--------------------------")

    test_cases = [
        ("Pothole", 0.90),
        ("Pothole", 0.70),
        ("Crack", 0.90),
        ("Crack", 0.70),
        ("Manhole", 0.90),
        ("Manhole", 0.70)
    ]

    for damage, confidence in test_cases:

        severity = calculate_severity(
            damage,
            confidence
        )

        print(
            f"{damage} | "
            f"Confidence: {confidence:.0%} | "
            f"Severity: {severity}"
        )