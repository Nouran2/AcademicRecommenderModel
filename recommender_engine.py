if not any(course_code.startswith(p) for p in allowed_prefixes):
                    continue
                
                if course_code in taken_courses:
                    continue
                
                score = float(final_scores[i])
                confidence_val = round((score / max_score) * 100, 1)
                
                recs.append({
                    "course_code": course_code,
                    "course_name": self.course_names[i],
                    "score": round(score, 4),
                    "confidence": f"{confidence_val}%"
                })

            # 5. منطق الـ Fallback: لو ملقاش مواد كفاية في تراك الطالب
            if len(recs) < 2:
                for i in range(len(self.course_codes)):
                    code = self.course_codes[i]
                    # تخطي لو المادة أخدها أو موجودة فعلاً في الترشيحات
                    if code in taken_courses or any(r["course_code"] == code for r in recs):
                        continue
                    
                    score = float(final_scores[i])
                    # استخدام max_score الموحد لمنع الـ Error
                    confidence_val = round((score / max_score) * 100, 1)
                    
                    recs.append({
                        "course_code": code, 
                        "course_name": self.course_names[i], 
                        "score": round(score, 4), 
                        "confidence": f"{confidence_val}%"
                    })
            
            # 6. الرد النهائي المرتب
            return {
                "dominant_track": dominant_track,
                "track_confidence": f"{round(track_conf_raw, 1)}%",
                "track_reasoning": f"Based on your profile, you show high alignment with {dominant_track}.",
                "recommendations": sorted(recs, key=lambda x: x["score"], reverse=True)[:3]
            }

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return {"error": str(e)}

    def retrain_model(self, data_url):
        from trainer import perform_training
        if perform_training(data_url, self.model_path):
            self._load_artifacts()
            return True
        return False
