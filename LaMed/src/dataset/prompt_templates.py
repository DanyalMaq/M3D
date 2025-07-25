Caption_templates = [
            "Can you provide a caption consists of findings for this medical image?",
            "Describe the findings of the medical image you see.",
            "Please caption this medical scan with findings.",
            "What is the findings of this image?",
            "Describe this medical scan with findings.",
            "Please write a caption consists of findings for this image.",
            "Can you summarize with findings the images presented?",
            "Please caption this scan with findings.",
            "Please provide a caption consists of findings for this medical image.",
            "Can you provide a summary consists of findings of this radiograph?",
            "What are the findings presented in this medical scan?",
            "Please write a caption consists of findings for this scan.",
            "Can you provide a description consists of findings of this medical scan?",
            "Please caption this medical scan with findings.",
            "Can you provide a caption consists of findings for this medical scan?",
            "Please generate a medical report based on this image.",
            "Can you generate a diagnose report from this image.",
            "Could you analyze and provide a caption for the findings in this medical image?",
            "Please describe the observations depicted in this medical scan.",
            "Can you summarize the findings of this image in a caption?",
            "What are the significant findings in this medical image?",
            "Please provide a detailed caption outlining the findings of this image.",
            "Could you interpret and describe the findings shown in this medical scan?",
            "What conclusions can you draw from the observations in this image?",
            "Please write a descriptive caption based on the findings in this scan.",
            "What key findings can you identify from examining this medical image?",
            "Could you generate a detailed report based on the observations in this image?",
            "Can you provide a diagnosis based on the findings in this image?",
            "Please generate a comprehensive report summarizing the findings in this image.",
            "Caption the findings in this medical image?",
            "Describe the findings you see.",
            "Caption this medical scan's findings.",
            "What are the findings here?",
            "Describe these findings.",
            "Summarize the findings in these images.",
            "Caption this scan's findings.",
            "Provide a caption for this medical image's findings.",
            "Summarize the findings of this radiograph.",
            "What findings are presented in this scan?",
            "Describe this scan's findings.",
            "Generate a medical report based on this image.",
            "Can you provide a diagnosis based on this image?",
]

Seg_template = [
    "What organ is shown in the masked region?",
    "Which organ is segmented in this image?",
    "What structure is highlighted by the mask?",
    "Identify the organ marked by the mask.",
    "What anatomical region does the mask correspond to?",
    "Which body part is covered by the highlighted area?",
    "What is the masked area indicating?",
    "What organ lies within the masked region?",
    "Can you name the organ selected by the segmentation mask?",
    "What organ is enclosed by the mask?",
    "Please identify the organ highlighted in the image.",
    "What does the mask represent in this scan?",
    "Tell me which organ is being segmented.",
    "Which anatomical structure has been masked?",
    "What region of the body does the mask refer to?"
]


PosREC_templates = {
"cls_questions": [
    "Can you find the {} in this image? Give coordinates.",
    "Can you find {} in this image? Please output the coordinates.",
    "Please bounding the {} by box in this image.",
    "Where is {} in this image? Please respond with a bounding box.",
    "Where is {} in this image? Please output the box.",
    "Can you locate the {} in this image? Please output its coordinates.",
    "Could you mark the {} by bounding box in this image?",
    "Where can I find the {} in this image? Please provide its bounding box.",
    "Identify the indicated {} in this image. Please provide the coordinates of its bounding box.",
    "Can you locate the {} in this image? Please provide coordinates.",
    "Where is the {} in this image? Please output its coordinates.",
    "Please outline the {} with a bounding box in this image.",
    "Identify the location of {} in this image. Provide the bounding box coordinates.",
    "Where can I find the {} in this image? Please give its bounding box.",
    "Locate the {} in this image. Output its coordinates, please.",
    "Could you mark the {} with a bounding box in this image?",
    "Spot the {} in this image and provide its bounding box coordinates.",
    "Please pinpoint the {} in this image. Output its coordinates.",
    "Identify the {} indicated in this image. Provide coordinates for its bounding box.",
],

"des_questions": [
    "Description: {} Please answer and find it by box based on the above description.",
    "Definition: {} Please answer and show the bounding box based on the above definition.",
    "Description: {} Can you answer and find it by coordinates based on the above description.",
    "Definition: {} Please output the bounding box and answer based on the above definition.",
    "Description: {} Respond and locate it using a bounding box according to the description.",
    "Definition: {} Please provide an answer and display the bounding box according to the given definition.",
    "Description: {} Can you identify and locate it by coordinates, following the provided description or definition?",
    "Definition: {} Please output the bounding box and provide an answer based on the provided definition.",
    "Based on the description or definition, please respond to {} and indicate its location with a bounding box.",
    "{} Please answer and find it by box based on the above description.",
    "{} Please answer and show the bounding box based on the above definition.",
    "{} Can you answer and find it by coordinates based on the above description.",
    "{} Please output the bounding box and answer based on the above definition.",
    "{} Respond and locate it using a bounding box according to the description.",
    "{} Please provide an answer and display the bounding box according to the given definition.",
    "{} Can you identify and locate it by coordinates, following the provided description or definition?",
    "{} Please output the bounding box and provide an answer based on the provided definition.",
    "Please answer and find it by box based on the description. {}",
    "Please answer and show the bounding box based on the definition. {}",
    "Can you answer and find it by coordinates based on the description. {}",
    "Please output the bounding box and answer based on the definition. {}",
    "Respond and locate it using a bounding box according to the description. {}",
    "Please provide an answer and display the bounding box according to the given definition. {}",
    "Can you identify and locate it by coordinates, following the provided description or definition? {}",
    "Please output the bounding box and provide an answer based on the provided definition. {}",
    "Description: {} Find and mark it with a bounding box.",
    "Definition: {} Show the bounding box and provide the answer.",
    "Describe: {} Identify and locate it by coordinates.",
    "Define: {} Output the bounding box and respond.",
    "Description: {} Locate it with a bounding box.",
    "Definition: {} Provide the answer with the bounding box.",
    "Description: {} Identify and locate it by coordinates.",
    "Definition: {} Output the bounding box with the answer.",
    "{} Answer and locate with a bounding box.",
    "{} Show the bounding box based on the definition.",
    "{} Locate it by coordinates according to the description.",
    "{} Output the bounding box and answer.",
    "{} Respond and locate with a bounding box.",
    "{} Provide the answer with the bounding box.",
    "{} Identify and locate by coordinates.",
    "{} Output the bounding box with the answer.",
],

"cls_answers": [
    "Coordinates are {}.",
    "Sure, {}.",
    "Sure, it is {}.",
    "Sure, the bounding box is {}.",
    "{}.",
    "Here are the coordinates: {}.",
    "Of course, it's located at {}.",
    "The bounding box is given by {}.",
    "The box is {}.",
    "Coordinates: {}.",
    "{}.",
    "Yes, {}.",
    "{} are the coordinates.",
    "Sure, it's at {}.",
    "The bounding box: {}.",
    "{} - here are the coordinates.",
    "{} - it's located at these coordinates.",
    "The box: {}.",
    "Coordinates: {}.",
    "Certainly, it's at {}.",
    "Yes, it is located at {}.",
    "Sure, the bounding box: {}.",
    "It's located at {}.",
    "Here are the coordinates: {}.",
    "Of course, it's positioned at {}.",
    "The bounding box can be described by: {}.",
    "The box is specified by: {}.",
    "Yes, the coordinates are {}.",
    "Sure, it's located around {}.",
    "Yes, it's within the bounding box: {}.",
    "Absolutely, the coordinates: {}.",
    "Yes, it's within the box: {}.",
],

"des_answers": [
    "The target is {} and the coordinates is {}.",
    "The category is {} and the bounding box is {}.",
    "It is {}, {}.",
    "{}, {}",
    "The target is identified as {} and its coordinates are {}.",
    "The category is {}, the bounding box is provided as {}.",
    "It is characterized by {}, with coordinates {}.",
    "The identified attributes are {}, {}.",
    "Describing it as {}, the corresponding box is {}.",
    "The target is {} and its coordinates are {}.",
    "Categorized as {}, with a bounding box of {}.",
    "It's {}, located at {}.",
    "It's characterized as {}, with coordinates {}.",
    "Identified as {}, with a bounding box: {}.",
    "Belongs to the category of {}, with coordinates {}.",
    "Characterized by {}, with a bounding box given as {}.",
    "Describing it as {}, with coordinates {}.",
    "Identified attributes: {}, with a bounding box of {}.",
    "The category is {}, and the coordinates are {}.",
    "Describing it as {}, its bounding box is {}.",
    "It falls under the category of {}, with coordinates {}.",
],

"cls_no_answers": [
    "Sorry, there is no {}.",
    "No, we can not see {}.",
    "{} is not here.",
    "Sorry, there's no {} in sight.",
    "Unfortunately, {} is not visible here.",
    "No, {} is not present in this image.",
    "Apologies, but there's no sign of {}.",
    "We couldn't find {} in this image.",
    "It seems {} is not included in this image.",
    "Sorry, no {} found.",
    "No {} visible.",
    "Not seeing {} here.",
    "{} absent.",
    "No {} detected.",
    "Can't find {} in this image.",
],

"des_no_answers": [
    "This is {}, but not here.",
    "This is {}, however we can not see it.",
    "This is {}, but it's not present here.",
    "This is {}, but unfortunately, it's not visible in this context.",
    "While this is {}, it doesn't seem to be here.",
    "This is {}, but it's not depicted in this image.",
    "While this is {}, it's not currently observable in this scene.",
    "{} mentioned, but not here.",
    "{} described, but not visible.",
    "This is {}, but it's not here.",
    "This is {}, but it's not visible.",
    "{} mentioned, but not depicted here.",
    "{} described, but not present.",
]
}


PosREG_templates = {
"cls_questions": [
            "What target is present within the coordinates {} ?",
            "Does the bounding box {} contain any target?",
            "Within the specified region {}, what target is present?",
            "Do you know what it is in the bounding box {}?",
            "What is it in this region {}?",
            "What object is located within the coordinates {}?",
            "Within the specified area {}, what object can be found?",
            "Can you identify the object within the bounding box {}?",
            "What object is present in this region {}?",
            "What's within coordinates {}?",
            "Does the box {} contain any target?",
            "What's in the region {}?",
            "What's within the box {}?",
            "What object is in this region {}?",
            "What's located within coordinates {}?",
            "What's within the area {}?",
            "Can you identify the object within box {}?",
            "What object is present in this area {}?",
            "What target can be found within coordinates {}?",
            "Is there any target within the bounding box {}?",
            "Within the specified region {}, what target is present?",
            "Can you identify what's within the bounding box {}?",
            "What object is present in this region {}?",
            "What's located within coordinates {}?",
            "Within the specified area {}, what object can be found?",
            "Could you identify the object within the bounding box {}?",
            "What object is present in this region {}?",
            "What's located at coordinates {}?",
            "Is there a target within the box {}?",
            "What's situated in the region {}?",
            "What's inside the box {}?",
            "What object can be found in this region {}?",
            "What's situated within coordinates {}?",
            "What's within the specified area {}?",
            "Can you identify any object within box {}?",
            "What object is present in this area {}?",
        ],

"des_questions": [
            "Please describe the target and its function based on the box {} in the image.",
            "Do you know what is it in this bounding box {}? Answer and explain it.",
            "What's the target in the bounding box {}? What function does it have?",
            "What is the area marked with a box {} in the image? Can you explain it?",
            "Could you describe the object and its purpose within the bounding box {} in the image?",
            "Can you identify and describe the object within this bounding box {}? Please explain.",
            "What is the object located in the bounding box {}? Could you explain its function?",
            "Could you describe the area outlined by the box {} in the image? Please explain its significance.",
            "Describe the target and its function in box {}.",
            "What's in this box {}? Describe and explain.",
            "What's in the box {}? Explain its function.",
            "What's inside the box {}? Can you describe it?",
            "Describe the object in box {} and its purpose.",
            "Identify and describe the object in box {}. Please explain.",
            "What's the object in box {}? Describe its function.",
            "Describe the area outlined by box {} and its significance.",
            "What's in the marked area {}? Please explain.",
        ],

"cls_answers": [
            "The target is {}.",
            "Sure, the bounding box contains {}.",
            "Sure, it is {}.",
            "Sure, {} is in the bounding box.",
            "{}.",
            "The object is {}.",
            "Of course, it's {}.",
            "Certainly, {} can be found in the bounding box.",
            "Yes, the bounding box includes {}.",
            "The target is identified as {}.",
            "Affirmative, the bounding box contains {}.",
            "Yes, it is indeed {}.",
            "Yes, {} is within the bounding box.",
            "The object in question is {}.",
            "Of course, it's indeed {}.",
            "Certainly, {} can be found within the bounding box.",
            "Yes, the bounding box includes {}.",
            "Indeed, {} is present within the bounding box.",
            "Confirmed, the object is {}.",
            "Absolutely, it's {}.",
        ],

"des_answers": [
            "Sure, it is {}. {}",
            "The category is {}. {}",
            "It is {}, {}",
            "{}, {}",
            "The target is identified as {} and its description is {}",
            "The category is {}. Description: {}",
            "It is characterized by {}, {}",
            "The identified attributes are {}, {}",
            "Sure, it is {}. Describing it as {}",
            "Certainly, it is {}. {}",
            "The category is identified as {}. {}",
            "It's labeled as {}, {}",
            "Identified as {}, {}",
            "The target is recognized as {} with the following description: {}",
            "The category is {}. Here's the description: {}",
            "Characterized by {}, {}",
            "Identified attributes: {}, {}",
            "Sure, it's {}. Described as {}",
        ],

"cls_no_answers": [
    "Sorry, there is no {}.",
    "No, we can not see {}.",
    "{} is not here.",
    "Sorry, we couldn't find any {}.",
    "Nope, {} isn't visible here.",
    "{} seems to be absent.",
    "Unfortunately, {} is not present.",
    "It appears there's no {} in sight.",
    "Regrettably, we couldn't detect any {}.",
],

"des_no_answers": [
    "This is {}, but not here.",
    "This is {}, however we can not see it.",
    "This is {}, but it's not present here.",
    "This is {}, however, it's not visible in this context.",
    "The answer is {}, but it is not here.",
    "The object is {}, but it is absent in the image.",
]
}


Seg_templates = {
"cls_questions": [
            "Can you segment the {} in this image?",
            "Can you segment {} in this image? Please output the mask.",
            "Please segment the {} in this image.",
            "What is {} in this image? Please respond with segmentation mask.",
            "What is {} in this image? Please output segmentation mask.",
            "Could you provide a segmentation for the {}?",
            "I need the {} segmented from this image.",
            "Segment {} from this image and provide the mask, please.",
            "Please provide a segmentation mask for the {} in this image.",
            "Can you identify and segment the {} in this image?",
            "Could you segment the {} here and share the mask?",
            "I'm looking for segmentation of {} in this image.",
            "Can you segment the object labeled as {}?",
            "Segment the {} in this image, please.",
            "Please segment {} in this image and provide the mask.",
            "What's the segmentation for {} in this image?",
            "Segment the {} in this image and share the mask.",
            "Please provide segmentation for {} in this image.",
            "Could you segment the {} here and share the mask?",
            "Segment the object labeled as {} and provide the mask.",
            "I need the {} segmented from this image. Can you help?",
            "Can you identify and segment the {} in this image?",
            "Can you segment {} here and output the mask?",
            "Please segment the {} in this image and share the mask.",
            "Segment {} from this image and provide the mask, please.",
            "Can you segment the {} in this image?",
        ],

"des_questions": [
            "{} Please answer and segment.",
            "{} Please output segmentation mask and answer.",
            "{} Please answer and segment based on the above description.",
            "{} Please answer and segment based on the above definition.",
            "{} Can you answer and segment it based on the above description or definition.",
            "{} Please output segmentation mask and answer based on the above description or definition.",
            "{} Please segment accordingly.",
            "{} Please provide segmentation and answer according to it.",
            "{} Now, segment it and provide your answer.",
            "{} Please segment and provide your response.",
            "{} Can you segment it accordingly?",
            "Description: {} Please answer and segment based on the above description.",
            "Definition: {} Please answer and segment based on the above definition.",
            "Description: {} Can you answer and segment it based on the above description or definition.",
            "Definition: {} Please output segmentation mask and answer based on the above description or definition.",
            "Provided description: {} Please segment accordingly.",
            "Given definition: {} Please provide segmentation and answer according to it.",
            "The description provided is: {} Now, segment it and provide your answer.",
            "Based on the provided definition: {} Please segment and provide your response.",
            "Describing the object as: {} Can you segment it accordingly?",
            "Defining it as: {} Now, segment and provide your answer.",
        ],

"cls_answers": [
            "It is [SEG].",
            "Sure, [SEG].",
            "Sure, it is [SEG].",
            "Sure, the segmentation result is [SEG].",
            "[SEG].",
            "The segmentation indicates [SEG].",
            "According to the segmentation, it is [SEG].",
            "The segmentation reveals [SEG].",
            "The segmentation suggests [SEG].",
            "From the segmentation, it appears to be [SEG].",
            "The target is [SEG].",
            "The segmentation mask is [SEG].",
            "The mask is [SEG].",
],

"des_answers": [
            "The target is {} and the segmentation mask is [SEG].",
            "The category is {} and the mask is [SEG].",
            "It is {}, [SEG].",
            "{}, [SEG]",
            "Identified as {}, here is the segmentation: [SEG].",
            "Categorized as {}, the segmentation is: [SEG].",
            "The class is {}, and the corresponding segmentation is: [SEG].",
            "Regarding the classification, it is {}, and the segmentation is: [SEG].",
            "Classified as {}, here's the segmentation: [SEG].",
            "The label assigned is {}, and the associated segmentation is: [SEG].",
            "Category: {}, segmentation: [SEG].",
            "It's classified as {}, with the segmentation: [SEG].",
        ],

"cls_no_answers": [
            "Sorry, there is no {}.",
            "No, we cannot see {}.",
            "{} is not present.",
            "There's no sign of {} in this image.",
            "Unfortunately, {} is not visible in this image.",
            "We cannot detect {} in this image.",
            "There's no indication of {} here.",
            "Regrettably, {} cannot be observed in this image.",
            "Sorry, {} isn't here.",
            "{} is absent.",
            "Nope, {} is missing.",
            "We're not seeing any {}.",
        ],

"des_no_answers": [
    "This is {}, but not here.",
    "This is {}, however we can not see it.",
    "While this is {}, it's not within view.",
    "This appears to be {}, but it's not in this vicinity.",
    "Although this is {}, it's not visible here.",
    "Though this is {}, it's not captured in this scene.",
    "This is indeed {}, but it's not here in the image.",
    "While we've identified {}, it's not within this image.",
    "This seems to be {}, but it's not depicted here.",
    "While this describes {}, it's not shown in this image.",
]
}
