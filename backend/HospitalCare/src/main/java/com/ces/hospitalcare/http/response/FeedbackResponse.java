package com.ces.hospitalcare.http.response;
import com.ces.hospitalcare.dto.UserDTO;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class FeedbackResponse {
  private UserDTO patient;

  private String commentText;
}
