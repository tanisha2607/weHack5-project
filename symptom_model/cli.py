import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "symptom_checker_model.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    artifact = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print(f"\nERROR: Could not find '{MODEL_FILE}'.")
    print("Make sure the .joblib file is in the same folder as cli.py.\n")
    exit()

model = artifact["model"]
ML_SYMPTOMS = artifact["ml_symptoms"]
RED_FLAG_SYMPTOMS = artifact["red_flag_symptoms"]
DURATION_THRESHOLDS = artifact["duration_thresholds"]
CONFIDENCE_THRESHOLD = artifact["confidence_threshold"]
SYMPTOM_ALIASES = artifact["symptom_aliases"]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_name(symptom):
    """
    Convert dataset-style symptom names into cleaner
    names for the terminal.

    This does NOT change the actual model feature name.
    """

    special_names = {
        "coryza": "Runny nose",
        "lacrimation": "Watery eyes",
        "regurgitation.1": "Regurgitation (variant)",
        "ache all over": "Body aches",
        "flu-like syndrome": "Flu-like symptoms",
        "itchiness of eye": "Itchy eyes",
        "skin dryness, peeling, scaliness, or roughness":
            "Dry / peeling / rough skin",
        "changes in stool appearance":
            "Changes in stool appearance",
        "symptoms of the face":
            "Facial symptoms",
        "abnormal appearing skin":
            "Abnormal skin appearance",
        "irregular appearing scalp":
            "Abnormal scalp appearance",
        "pain of the anus":
            "Anal pain",
    }

    if symptom in special_names:
        return special_names[symptom]

    return symptom.capitalize()


def print_header(title):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)


# ============================================================
# SYMPTOM SELECTION
# ============================================================

def select_symptoms():

    print_header("AI SYMPTOM CHECKER")

    print("\nSelect all symptoms that apply.")
    print("Enter their numbers separated by commas.")
    print("Example: 4, 7, 8")
    print("Type 'q' to quit.\n")

    # Display symptoms in two columns
    for i in range(0, len(ML_SYMPTOMS), 2):

        left_number = i + 1
        left_name = display_name(ML_SYMPTOMS[i])

        left_text = f"{left_number:>2}. {left_name:<38}"

        if i + 1 < len(ML_SYMPTOMS):
            right_number = i + 2
            right_name = display_name(ML_SYMPTOMS[i + 1])

            right_text = f"{right_number:>2}. {right_name}"

            print(left_text + right_text)
        else:
            print(left_text)

    print("\n------------------------------------------------------------")

    while True:

        choice = input("Enter symptom numbers: ").strip().lower()

        if choice == "q":
            return None

        if not choice:
            print("Please select at least one symptom.")
            continue

        try:
            numbers = [
                int(x.strip())
                for x in choice.split(",")
            ]
        except ValueError:
            print("Please enter numbers separated by commas.")
            continue

        # Remove duplicates while preserving order
        numbers = list(dict.fromkeys(numbers))

        invalid = [
            n for n in numbers
            if n < 1 or n > len(ML_SYMPTOMS)
        ]

        if invalid:
            print(
                f"Invalid number(s): {invalid}. "
                f"Choose between 1 and {len(ML_SYMPTOMS)}."
            )
            continue

        selected = [
            ML_SYMPTOMS[n - 1]
            for n in numbers
        ]

        return selected


# ============================================================
# DURATION INPUT
# ============================================================

def get_duration():

    while True:

        value = input(
            "\nHow many days have the symptoms lasted? "
        ).strip()

        try:
            days = int(value)

            if days < 0:
                print("Please enter 0 or a positive number.")
                continue

            return days

        except ValueError:
            print("Please enter a whole number.")


# ============================================================
# INFERENCE
# ============================================================

def predict_condition(symptoms, duration_days):

    # --------------------------------------------------------
    # RED FLAG CHECK
    # --------------------------------------------------------

    red_flags_found = [
        symptom
        for symptom in symptoms
        if symptom in RED_FLAG_SYMPTOMS
    ]

    if red_flags_found:

        return {
            "prediction": "Medical evaluation recommended",
            "confidence": 1.0,
            "recommend_doctor": True,
            "reason": "red_flag"
        }

    # --------------------------------------------------------
    # CREATE INPUT VECTOR
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=ML_SYMPTOMS
    )

    # --------------------------------------------------------
    # TURN SELECTED SYMPTOMS ON
    # --------------------------------------------------------

    for symptom in symptoms:

        if symptom in input_data.columns:
            input_data.loc[0, symptom] = 1

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    confidence = float(probabilities.max())

    # --------------------------------------------------------
    # SAFETY RULES
    # --------------------------------------------------------

    recommend_doctor = False

    reason = None

    # Low confidence
    if confidence < CONFIDENCE_THRESHOLD:

        recommend_doctor = True
        reason = "low_confidence"

    # Duration threshold
    threshold = DURATION_THRESHOLDS.get(prediction)

    if threshold is not None and duration_days > threshold:

        recommend_doctor = True

        if reason is None:
            reason = "duration"

        else:
            reason = "low_confidence_and_duration"

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": confidence,
        "recommend_doctor": recommend_doctor,
        "reason": reason
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result, duration_days):

    print_header("RESULT")

    prediction = result["prediction"]
    confidence = result["confidence"]
    recommend_doctor = result["recommend_doctor"]
    reason = result["reason"]

    # Red flag result
    if reason == "red_flag":

        print("\n⚠  MEDICAL EVALUATION RECOMMENDED")
        print("\nA red-flag symptom was selected.")
        print(
            "The normal condition prediction has been "
            "overridden for safety."
        )
        print(
            "\nPlease seek appropriate medical evaluation."
        )

        print("\n" + "-" * 60)
        return

    # Normal ML result
    print(f"\nLikely condition : {prediction}")
    print(f"Model confidence : {confidence * 100:.1f}%")

    threshold = DURATION_THRESHOLDS.get(prediction)

    if threshold is not None:
        print(f"Duration entered  : {duration_days} day(s)")
        print(f"Prototype threshold: {threshold} day(s)")

    print("\n------------------------------------------------------------")

    if recommend_doctor:

        print("⚠  RECOMMENDATION: SEE A DOCTOR")

        if reason == "low_confidence":
            print(
                "\nThe model could not confidently classify "
                "these symptoms."
            )

        elif reason == "duration":
            print(
                "\nThe symptoms have lasted longer than "
                "the configured prototype threshold."
            )

        elif reason == "low_confidence_and_duration":
            print(
                "\nThe model confidence is low and the symptoms "
                "have exceeded the configured duration threshold."
            )

        print(
            "\nPlease consider speaking with a healthcare "
            "professional."
        )

    else:

        print("✓ No doctor recommendation triggered")
        print(
            "\nThis result is intended only as a prototype "
            "symptom-screening aid."
        )

    print("\n" + "-" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("AI SYMPTOM CHECKER".center(60))
    print("Hackathon Prototype".center(60))
    print("=" * 60)

    print("\nModel loaded successfully.")
    print(f"Available symptoms: {len(ML_SYMPTOMS)}")
    print(f"Supported conditions: {len(model.classes_)}")

    while True:

        symptoms = select_symptoms()

        if symptoms is None:
            break

        duration_days = get_duration()

        result = predict_condition(
            symptoms,
            duration_days
        )

        print_result(
            result,
            duration_days
        )

        print("\nWould you like to run another check?")
        again = input("Enter y to continue or n to quit: ").strip().lower()

        if again != "y":
            break

    print("\nThank you for using the AI Symptom Checker.")
    print("Stay safe!\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()