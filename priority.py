"""Road Sense - Road Repair Priority Module"""


def calculate_priority(severity):
    """
    Convert severity level into repair priority.

    Parameters:
        severity: Severity level of detected damage.

    Returns:
        Repair priority.
    """

    severity = severity.lower()

    if severity == "critical":
        return "Immediate"

    elif severity == "high":
        return "High"

    elif severity == "medium":
        return "Moderate"

    elif severity == "low":
        return "Low"

    else:
        return "Unknown"


# --------------------------------------------------
# TEST PRIORITY MODULE
# --------------------------------------------------

if __name__ == "__main__":

    print("Road Sense Priority Module")
    print("--------------------------")

    test_severities = [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]

    for severity in test_severities:

        priority = calculate_priority(severity)

        print(
            f"Severity: {severity} "
            f"| Priority: {priority}"
        )