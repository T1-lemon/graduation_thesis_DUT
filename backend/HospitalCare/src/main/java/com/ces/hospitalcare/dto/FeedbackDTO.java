package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class FeedbackDTO extends BaseDTO {
  private UserDTO patient;

  private String commentText;
}
