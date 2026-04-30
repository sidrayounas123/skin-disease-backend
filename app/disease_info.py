DISEASE_INFO = {
    "AD (Atopic Dermatitis)": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "Chronic inflammatory skin condition causing dry, itchy, and inflamed skin",
        "precautions": [
            "Moisturize skin regularly with fragrance-free creams",
            "Avoid known allergens and irritants",
            "Use gentle, unscented soaps and detergents",
            "Wear soft, breathable cotton clothing",
            "Maintain consistent indoor humidity levels",
            "Avoid extreme temperature changes"
        ],
        "initial_treatment": [
            "Apply topical corticosteroids as prescribed",
            "Use moisturizers containing ceramides",
            "Take antihistamines for severe itching",
            "Use cool compresses to reduce inflammation"
        ],
        "see_doctor": True
    },
    "CD (Contact Dermatitis)": {
        "severity": "Moderate",
        "severity_score": 5,
        "description": "Skin inflammation caused by direct contact with allergens or irritants",
        "precautions": [
            "Identify and avoid triggering substances",
            "Wear protective gloves when handling chemicals",
            "Use barrier creams before potential exposure",
            "Test new products on small skin areas first",
            "Keep skin clean and dry",
            "Use hypoallergenic personal care products"
        ],
        "initial_treatment": [
            "Wash affected area with mild soap and water",
            "Apply calamine lotion for itching",
            "Use cold compresses to reduce swelling",
            "Apply over-the-counter hydrocortisone cream"
        ],
        "see_doctor": True
    },
    "EC (Eczema)": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "Chronic condition causing dry, itchy, inflamed skin patches",
        "precautions": [
            "Keep skin well-moisturized",
            "Avoid harsh soaps and hot water",
            "Use fragrance-free products",
            "Wear soft, non-irritating clothing",
            "Manage stress levels",
            "Avoid known triggers"
        ],
        "initial_treatment": [
            "Apply emollients regularly",
            "Use topical corticosteroids",
            "Take antihistamines for itching",
            "Use wet wrap therapy for severe cases"
        ],
        "see_doctor": True
    },
    "SC (Scabies)": {
        "severity": "Moderate",
        "severity_score": 5,
        "description": "Contagious skin infestation caused by mites burrowing into skin",
        "precautions": [
            "Avoid close physical contact with infected individuals",
            "Do not share personal items (clothing, towels)",
            "Wash bedding and clothing in hot water",
            "Vacuum furniture and carpets thoroughly",
            "Isolate affected family members",
            "Practice good personal hygiene"
        ],
        "initial_treatment": [
            "Apply prescription scabicide medications",
            "Treat all family members simultaneously",
            "Wash all bedding and clothing in hot water",
            "Use calamine lotion for itching"
        ],
        "see_doctor": True
    },
    "SD (Seborrheic Dermatitis)": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "Inflammatory condition causing flaky, itchy skin on oily areas",
        "precautions": [
            "Keep skin clean but avoid over-washing",
            "Use medicated shampoos for scalp involvement",
            "Manage stress levels",
            "Avoid harsh skin products",
            "Protect skin from extreme weather",
            "Maintain healthy diet and lifestyle"
        ],
        "initial_treatment": [
            "Use antifungal creams",
            "Apply topical corticosteroids",
            "Use medicated shampoos with ketoconazole",
            "Apply gentle moisturizers"
        ],
        "see_doctor": True
    },
    "TC (Tinea Corporis)": {
        "severity": "Mild",
        "severity_score": 3,
        "description": "Fungal infection causing ring-shaped rash on body",
        "precautions": [
            "Keep skin clean and dry",
            "Avoid sharing personal items",
            "Wear breathable clothing",
            "Treat infected pets promptly",
            "Use separate towels for affected areas",
            "Avoid walking barefoot in shared areas"
        ],
        "initial_treatment": [
            "Apply over-the-counter antifungal creams",
            "Keep affected area clean and dry",
            "Use antifungal powders in shoes and socks",
            "Continue treatment for 2-4 weeks after clearance"
        ],
        "see_doctor": False
    },
    "Acne": {
        "severity": "Mild",
        "severity_score": 3,
        "description": "Common skin condition causing pimples and spots",
        "precautions": [
            "Wash face twice daily with gentle cleanser",
            "Avoid touching or picking at pimples",
            "Use oil-free, non-comedogenic products",
            "Change pillowcases frequently",
            "Stay hydrated and avoid oily foods",
            "Remove makeup before sleeping"
        ],
        "initial_treatment": [
            "Apply salicylic acid or benzoyl peroxide cleanser",
            "Use oil-free moisturizer",
            "Apply spot treatment on affected areas",
            "Use non-comedogenic sunscreen"
        ],
        "see_doctor": False
    },
    "Actinic Keratosis": {
        "severity": "Severe",
        "severity_score": 7,
        "description": "Rough, scaly patches caused by sun damage, potentially precancerous",
        "precautions": [
            "Use broad-spectrum sunscreen daily",
            "Avoid peak sun hours (10am-4pm)",
            "Wear protective clothing and hats",
            "Regular skin self-examinations",
            "Avoid tanning beds",
            "Schedule regular dermatologist visits"
        ],
        "initial_treatment": [
            "Apply prescribed topical medications",
            "Use cryotherapy for individual lesions",
            "Consider photodynamic therapy",
            "Use chemical peels as recommended"
        ],
        "see_doctor": True
    },
    "Basal Cell Carcinoma": {
        "severity": "Severe",
        "severity_score": 8,
        "description": "Most common type of skin cancer, slow-growing but locally destructive",
        "precautions": [
            "Daily sunscreen use with SPF 30+",
            "Regular skin examinations",
            "Avoid excessive sun exposure",
            "Protect skin with clothing and hats",
            "Early detection through self-checks",
            "Annual dermatologist visits"
        ],
        "initial_treatment": [
            "Immediate dermatologist consultation required",
            "Surgical excision is primary treatment",
            "Mohs surgery for facial lesions",
            "Radiation therapy for advanced cases"
        ],
        "see_doctor": True
    },
    "Chickenpox": {
        "severity": "Moderate",
        "severity_score": 5,
        "description": "Highly contagious viral infection causing itchy blister rash",
        "precautions": [
            "Isolate infected person",
            "Avoid scratching blisters",
            "Keep fingernails trimmed short",
            "Use separate towels and bedding",
            "Avoid contact with pregnant women",
            "Stay home until all blisters crust over"
        ],
        "initial_treatment": [
            "Calamine lotion for itching",
            "Cool baths with baking soda",
            "Acetaminophen for fever",
            "Antihistamines for severe itching"
        ],
        "see_doctor": True
    },
    "Dermato Fibroma": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "Benign fibrous growth in the skin, typically firm and painless",
        "precautions": [
            "Monitor for changes in size or appearance",
            "Avoid trauma to the area",
            "Regular self-examinations",
            "Protect from sun exposure",
            "Document growth patterns",
            "Avoid attempting removal at home"
        ],
        "initial_treatment": [
            "Surgical excision if bothersome",
            "Cryotherapy for small lesions",
            "Laser treatment for cosmetic removal",
            "Regular monitoring for changes"
        ],
        "see_doctor": True
    },
    "Dyshidrotic Eczema": {
        "severity": "Moderate",
        "severity_score": 5,
        "description": "Recurring condition causing small, intensely itchy blisters on hands and feet",
        "precautions": [
            "Keep hands and feet dry",
            "Avoid contact with irritants",
            "Wear protective gloves",
            "Use gentle soaps and moisturizers",
            "Manage stress levels",
            "Avoid extreme temperatures"
        ],
        "initial_treatment": [
            "Apply topical corticosteroids",
            "Use cold compresses for itching",
            "Apply barrier creams",
            "Take oral antihistamines"
        ],
        "see_doctor": True
    },
    "Melanoma": {
        "severity": "Severe",
        "severity_score": 9,
        "description": "Most serious type of skin cancer, can spread rapidly if untreated",
        "precautions": [
            "Regular skin self-examinations",
            "ABCDE rule monitoring (Asymmetry, Border, Color, Diameter, Evolving)",
            "Daily sunscreen use",
            "Avoid tanning beds",
            "Regular dermatologist check-ups",
            "Protect skin from UV radiation"
        ],
        "initial_treatment": [
            "Immediate medical attention required",
            "Surgical excision with wide margins",
            "Sentinel lymph node biopsy",
            "Immunotherapy for advanced cases"
        ],
        "see_doctor": True
    },
    "Nail Fungus": {
        "severity": "Mild",
        "severity_score": 2,
        "description": "Fungal infection affecting toenails and fingernails",
        "precautions": [
            "Keep feet clean and dry",
            "Wear breathable shoes and socks",
            "Avoid walking barefoot in public areas",
            "Use separate nail clippers",
            "Disinfect shoes regularly",
            "Avoid sharing nail tools"
        ],
        "initial_treatment": [
            "Apply topical antifungal medications",
            "Use antifungal nail polishes",
            "Keep nails trimmed short",
            "Consider oral antifungal medications"
        ],
        "see_doctor": False
    },
    "Nevus": {
        "severity": "Mild",
        "severity_score": 1,
        "description": "Common benign growth of melanocytes (moles)",
        "precautions": [
            "Monitor for changes in size, shape, or color",
            "Regular self-examinations",
            "Protect from sun exposure",
            "Avoid irritation or trauma",
            "Document with photographs",
            "Regular dermatologist check-ups"
        ],
        "initial_treatment": [
            "Usually no treatment needed",
            "Surgical removal if suspicious or bothersome",
            "Regular monitoring for changes",
            "Biopsy if concerning features develop"
        ],
        "see_doctor": False
    },
    "Normal Skin": {
        "severity": "Mild",
        "severity_score": 0,
        "description": "Healthy skin without any disease or abnormal conditions",
        "precautions": [
            "Maintain regular skincare routine",
            "Use sunscreen daily",
            "Stay hydrated",
            "Eat balanced diet",
            "Get adequate sleep",
            "Avoid excessive sun exposure"
        ],
        "initial_treatment": [
            "Continue regular skincare",
            "Use gentle cleansers",
            "Apply moisturizer as needed",
            "Use sunscreen with SPF 30+"
        ],
        "see_doctor": False
    },
    "Pigmented Benign Keratosis": {
        "severity": "Moderate",
        "severity_score": 6,
        "description": "Benign growth with wart-like appearance and stuck-on appearance",
        "precautions": [
            "Regular monitoring for changes",
            "Sun protection",
            "Avoid picking or scratching",
            "Regular dermatologist visits",
            "Document growth patterns",
            "Avoid self-treatment"
        ],
        "initial_treatment": [
            "Cryotherapy for removal",
            "Curettage and electrodesiccation",
            "Laser treatment options",
            "Topical retinoids"
        ],
        "see_doctor": True
    },
    "Ringworm": {
        "severity": "Mild",
        "severity_score": 3,
        "description": "Fungal infection causing ring-shaped rash on skin",
        "precautions": [
            "Keep skin clean and dry",
            "Avoid sharing personal items",
            "Treat infected pets",
            "Wear breathable clothing",
            "Use separate towels",
            "Avoid walking barefoot in shared areas"
        ],
        "initial_treatment": [
            "Apply topical antifungal creams",
            "Keep area clean and dry",
            "Continue treatment 2 weeks after clearance",
            "Use antifungal powders in shoes"
        ],
        "see_doctor": False
    },
    "Seborrheic Keratosis": {
        "severity": "Severe",
        "severity_score": 7,
        "description": "Benign skin growth with waxy, scaly appearance",
        "precautions": [
            "Sun protection",
            "Regular monitoring for changes",
            "Avoid picking or scratching",
            "Regular dermatologist check-ups",
            "Document any changes",
            "Gentle skin care"
        ],
        "initial_treatment": [
            "Cryotherapy for removal",
            "Curettage",
            "Laser treatment",
            "Chemical peels"
        ],
        "see_doctor": True
    },
    "Squamous Cell Carcinoma": {
        "severity": "Severe",
        "severity_score": 8,
        "description": "Common form of skin cancer that can spread if untreated",
        "precautions": [
            "Daily sunscreen use",
            "Regular skin examinations",
            "Avoid excessive sun exposure",
            "Protect skin with clothing",
            "Early detection through self-checks",
            "Regular dermatologist visits"
        ],
        "initial_treatment": [
            "Immediate medical attention required",
            "Surgical excision",
            "Mohs surgery for facial lesions",
            "Radiation therapy for advanced cases"
        ],
        "see_doctor": True
    },
    "Vascular Lesion": {
        "severity": "Severe",
        "severity_score": 7,
        "description": "Abnormal growth of blood vessels in the skin",
        "precautions": [
            "Avoid trauma to the area",
            "Regular monitoring for changes",
            "Sun protection",
            "Avoid picking or scratching",
            "Regular medical follow-up",
            "Document growth patterns"
        ],
        "initial_treatment": [
            "Laser treatment for removal",
            "Sclerotherapy injections",
            "Cryotherapy for small lesions",
            "Surgical excision for large lesions"
        ],
        "see_doctor": True
    },
    "Eczema": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "Chronic condition causing dry, itchy, inflamed skin patches",
        "precautions": [
            "Keep skin well-moisturized",
            "Avoid harsh soaps and hot water",
            "Use fragrance-free products",
            "Wear soft, non-irritating clothing",
            "Manage stress levels",
            "Avoid known triggers"
        ],
        "initial_treatment": [
            "Apply emollients regularly",
            "Use topical corticosteroids",
            "Take antihistamines for itching",
            "Use wet wrap therapy for severe cases"
        ],
        "see_doctor": True
    },
    "Psoriasis": {
        "severity": "Moderate",
        "severity_score": 5,
        "description": "Autoimmune condition causing red, scaly patches on skin",
        "precautions": [
            "Keep skin moisturized",
            "Avoid skin injuries and scratches",
            "Manage stress levels",
            "Avoid alcohol and smoking",
            "Maintain healthy weight",
            "Protect skin from extreme weather"
        ],
        "initial_treatment": [
            "Use topical corticosteroids",
            "Apply vitamin D analogues",
            "Use phototherapy as prescribed",
            "Take systemic medications for severe cases"
        ],
        "see_doctor": True
    },
    "Rosacea": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "Chronic skin condition causing redness and visible blood vessels",
        "precautions": [
            "Avoid sun exposure and extreme temperatures",
            "Identify and avoid triggers (spicy foods, alcohol)",
            "Use gentle skincare products",
            "Protect skin with broad-spectrum sunscreen",
            "Avoid harsh scrubbing",
            "Manage stress levels"
        ],
        "initial_treatment": [
            "Use topical antibiotics or metronidazole",
            "Apply brimonidine gel for redness",
            "Take oral antibiotics for inflammatory cases",
            "Use gentle, non-irritating cleansers"
        ],
        "see_doctor": True
    },
    "Fungal Infection": {
        "severity": "Mild",
        "severity_score": 3,
        "description": "Infection caused by fungi affecting skin, hair, or nails",
        "precautions": [
            "Keep skin clean and dry",
            "Avoid sharing personal items",
            "Wear breathable clothing",
            "Use separate towels for affected areas",
            "Treat infected pets promptly",
            "Avoid walking barefoot in shared areas"
        ],
        "initial_treatment": [
            "Apply over-the-counter antifungal creams",
            "Keep affected area clean and dry",
            "Use antifungal powders in shoes and socks",
            "Continue treatment for 2-4 weeks after clearance"
        ],
        "see_doctor": False
    },
    "Bacterial Infection": {
        "severity": "Moderate",
        "severity_score": 5,
        "description": "Infection caused by bacteria entering skin through cuts or breaks",
        "precautions": [
            "Keep wounds clean and covered",
            "Practice good hand hygiene",
            "Avoid sharing personal items",
            "Treat cuts and scrapes promptly",
            "Avoid scratching affected areas",
            "Keep skin moisturized to prevent cracks"
        ],
        "initial_treatment": [
            "Clean area with antiseptic solution",
            "Apply antibiotic ointment",
            "Keep area covered with clean bandage",
            "Take oral antibiotics for severe infections"
        ],
        "see_doctor": True
    },
    "Allergic Reaction": {
        "severity": "Mild",
        "severity_score": 3,
        "description": "Skin reaction to allergens causing rash, itching, or swelling",
        "precautions": [
            "Identify and avoid known allergens",
            "Use hypoallergenic products",
            "Test new products on small skin areas",
            "Keep skin moisturized",
            "Wear protective clothing",
            "Maintain detailed allergy diary"
        ],
        "initial_treatment": [
            "Apply calamine lotion for itching",
            "Take oral antihistamines",
            "Use cold compresses for swelling",
            "Apply hydrocortisone cream for inflammation"
        ],
        "see_doctor": False
    },
    "Dermatitis": {
        "severity": "Moderate",
        "severity_score": 4,
        "description": "General inflammation of skin causing redness, itching, and rash",
        "precautions": [
            "Avoid known irritants and allergens",
            "Use gentle, fragrance-free products",
            "Keep skin well-moisturized",
            "Wear soft, breathable clothing",
            "Avoid excessive washing",
            "Protect skin from extreme temperatures"
        ],
        "initial_treatment": [
            "Apply topical corticosteroids",
            "Use moisturizers containing ceramides",
            "Take antihistamines for itching",
            "Use cool compresses to reduce inflammation"
        ],
        "see_doctor": True
    }
}
