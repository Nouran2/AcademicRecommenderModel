def build_dynamic_courses(catalog_data):
    # خريطة الأوزان الفريدة (Fingerprints) لمنع التساوي
    category_map = {
        "Artificial Intelligence": [0.0, 1.0, 0.0, 0.0], # AI صريح
        "Information Technology":  [0.0, 0.0, 1.0, 0.0], # IT صريح
        "Information Systems":     [0.0, 0.0, 0.0, 1.0], # IS صريح
        "Software Engineering":    [1.0, 0.0, 0.0, 0.0], # Prog صريح
        
        # كليات خارجية بأوزان "فريدة" ومائلة لتراكاتنا
        "Business Administration": [0.01, 0.01, 0.01, 0.9], # مائل بقوة للـ IS (نظم معلومات إدارية)
        "Faculty of Medicine":     [0.01, 0.4, 0.01, 0.2],  # مائل للـ AI (عشان الـ Bioinformatics)
        "Faculty of Arts":         [0.1, 0.01, 0.01, 0.4],  # مائل للـ IS و Programming (عشان الـ UI/UX)
        "Engineering":             [0.2, 0.1, 0.5, 0.1]     # مائل للـ IT (عشان الدوائر والشبكات)
    }
    # ... باقي الكود كما هو
