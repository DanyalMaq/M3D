from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "Qwen/Qwen3-8B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Input example
gt_preds = ["""Ground Truth
Region: Right level 1b node
Anatomy: PET/CT axial slice 71
Finding: Moderate to intensely FDG avid right level 1b node
SUV Max: 4.6
Axial Slice: 71
Prediction
Region: Level IIb
Anatomy: Lymph node
Finding: A 9 mm level IIb lymph node is present with a max SUV of 4.9 at slice 69.
SUV Max: 4.9
Axial Slice: 69
""",
"""
Ground Truth
Region: N/A
Anatomy: Retrocaval
Finding: an intensely FDG avid 2.5 x 2.2 cm retrocaval node measuring SUV max of 18.2 (PET/CT axial slice 176)
SUV Max: 18.2
Axial Slice: 176
Prediction
Region: Retroperitoneal
Anatomy: Lymph nodes
Finding: There are multiple enlarged retroperitoneal lymph nodes, the largest measuring 2.3 x 1.8 cm with an SUV max of 16.3 (PET/CT axial slice 187).
SUV Max: 16.3
Axial Slice: 187
""",
"""
Ground Truth
Region: Left thyroid
Anatomy: Thyroid
Finding: There is redemonstration of an intensely hypermetabolic left thyroid nodule which has been previously biopsied and shown to be benign.
SUV Max: N/A
Axial Slice: 89
Prediction
Region: Head/Neck
Anatomy: Left thyroid lobe
Finding: There is a focus of intense FDG uptake fusing to the left thyroid lobe that shows SUV max of 7.1 (axial PET/CT slice 85) and likely represents a metastatic lesion.
SUV Max: 7.1
Axial Slice: 85
""",
"""
Ground Truth
Region: Left paravertebral musculature
Anatomy: Upper neck
Finding: Increased FDG uptake within the left paravertebral musculature in the upper neck
SUV Max: 3.5
Axial Slice: 62
Prediction
Region: Left parotid gland
Anatomy: Parotid gland
Finding: There is a 1.5 cm soft tissue nodule within the posterior aspect of the left parotid gland with mild to moderate FDG uptake, with SUV max 3.4, best seen on PET/CT axial slice 54.
SUV Max: 3.4
Axial Slice: 54
""",
"""
Ground Truth
Region: Right inguinal lymph node
Anatomy: PET/CT axial slice 285
Finding: There is demonstration of an approximately 0.8 cm short axis moderately FDG avid and mildly prominent appearing right inguinal lymph node with SUV max of 4.8 (PET/CT axial slice 285).
SUV Max: 4.8
Axial Slice: 285
Prediction
Region: Right inguinal lymph node
Anatomy: PET/CT axial slice 252
Finding: -Right inguinal lymph node, 1.2 x 0.8 cm, SUV max 5.2 (PET/CT axial slice 252)
SUV Max: 5.2
Axial Slice: 252
""",
"""
Ground Truth
Region: Soft tissue
Anatomy: Left external iliac vein
Finding: However, there is a new area of soft tissue thickening with associated intense FDG uptake just medial to the left external iliac vein, extending inferiorly and anteriorly towards the space of Retzius
SUV Max: 11.0
Axial Slice: 234
Prediction
Region: Left external iliac lymph node
Anatomy: N/A
Finding: There is an intensely hypermetabolic left external iliac lymph node.
SUV Max: 12.3
Axial Slice: 244
""",
"""
Ground Truth
Region: Nasal bridge
Anatomy: superior aspect
Finding: Focal intense FDG uptake is seen within the superior aspect of the nasal bridge.
SUV Max: 6.8
Axial Slice: 55
Prediction
Region: N/A
Anatomy: Right maxillary alveolar process
Finding: There is an additional focus of intense FDG uptake fusing to the right maxillary alveolar process with corresponding erosive changes on CT.
SUV Max: N/A
Axial Slice: 59
""",
"""
Ground Truth
Region: N/A
Anatomy: Soft tissue nodule, right renal vein
Finding: A third FDG avid soft tissue nodule is located just anterior to the right renal vein and measures 2.5 x 2.0 with a max SUV of 12.6 on axial slice 155.
SUV Max: 12.6
Axial Slice: 155
Prediction
Region: Abdomen/Pelvis
Anatomy: Gastric wall
Finding: There is a large gastric mass with intense FDG uptake (PET/CT axial slice 166).
SUV Max: N/A
Axial Slice: 166
""",
"""
Ground Truth
Region: Aortopulmonary lymph node
Anatomy: PET/CT axial slice 120
Finding: Moderate to intensely FDG avid aortopulmonary lymph node with SUV max of 6.4 (PET/CT axial slice 120), previously it was 2.8.
SUV Max: 6.4
Axial Slice: 120
Prediction
Region: Left lower paratracheal lymph node
Anatomy: PET/CT axial slice 106
Finding: - Left lower paratracheal lymph node, 1.0 x 0.7 cm, SUV max 5.9 (PET/CT axial slice 106).
SUV Max: 5.9
Axial Slice: 106
""",
"""
Ground Truth
Region: Right
Anatomy: Internal mammary lymph nodes
Finding: There is a small lymph node on the right just lateral to the internal mammary lymph nodes,
SUV Max: 2.9
Axial Slice: 116
Prediction
Region: Right internal mammary lymph node
Anatomy: Lymph node
Finding: There is a mildly hypermetabolic right internal mammary lymph node (slice 117) that shows SUV max of 3.4.
SUV Max: 3.4
Axial Slice: 117"""]
result = []

def compute_overall_average(strings):
    numbers = []

    for s in strings:
        match = re.search(r"Overall:\s*([0-5](?:\.\d+)?)", s)
        if match:
            numbers.append(float(match.group(1)))

    if numbers:
        average = sum(numbers) / len(numbers)
        return average
    else:
        return None  # or 0.0 if you prefer a default

def eval(gt_pred):
    global result
    # Prompt construction
    user_prompt = f"""
    You are a radiologist evaluating the similarity of two findings in a PET/CT report: one is the ground truth, and the other is an AI model's prediction.

    Evaluate the similarity between the Ground Truth and Prediction based on:
    1. **Anatomic Region Match (1-5)** - Are the described regions close enough clinically?
    2. **Description of Findings (1-5)** - Are the descriptions of the lymph nodes (e.g., size, type, avidity) semantically similar?
    3. **Numerical Agreement (1-5)** - Are size, SUV max, and slice numbers reasonably close?

    Each rating should be from 1 (very different) to 5 (almost identical). Provide a short rationale for each. Give a final overall average score.

    {gt_pred}

    Respond with ratings in this format:
    Anatomic Region Match: X - [reason]
    Finding Description Match: X - [reason]
    Numerical Agreement: X - [reason]
    Overall: X
    """

    # Format as chat message
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    # Format input for chat model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize and move to device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )

    # Decode output
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # Display result
    print("Model Evaluation:\n")
    print(response)
    result.append(response)

for gt_pred in gt_preds:
    eval(gt_pred)

average = compute_overall_average(result)
print(f"Average Overall Score: {average:.2f}")
