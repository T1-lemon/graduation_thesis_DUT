package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.FeedbackDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.request.FeedbackRequest;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.IFeedbackService;
import com.ces.hospitalcare.service.IUserService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class FeedBackController {
  @Autowired
  private IUserService userService;

  @Autowired
  private IFeedbackService feedbackService;

  @GetMapping(value = "/guest/feedbacks")
  public List<FeedbackDTO> getAllFeedbackByExaminationId(
      @RequestParam("examinationId") Long medicalExaminationId) {

    return feedbackService.getAllFeedbackByExaminationId(medicalExaminationId);
  }

  @PostMapping(path = "/feedbacks")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_PATIENT')")
  public FeedbackDTO addFeedbackByPatient(
      @RequestBody FeedbackRequest feedbackRequest) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO patient = userResponse.getUser();
    feedbackRequest.setPatientId(patient.getId());
    return feedbackService.addFeedbackByPatient(feedbackRequest);
  }
}
