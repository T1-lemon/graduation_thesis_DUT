package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.FeedbackDTO;
import com.ces.hospitalcare.http.request.FeedbackRequest;
import java.util.List;

public interface IFeedbackService {
  List<FeedbackDTO> getAllFeedbackByExaminationId(Long examinationId);

  FeedbackDTO addFeedbackByPatient(FeedbackRequest feedbackRequest);
}
