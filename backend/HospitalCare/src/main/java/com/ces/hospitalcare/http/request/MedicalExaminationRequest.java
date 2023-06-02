package com.ces.hospitalcare.http.request;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class MedicalExaminationRequest {
  private Long examinationPrice;

  private String title;

  private String shortDescription;

  private String description;

  private Long departmentId;

  private Long doctorId;
}
