package com.ces.hospitalcare.http.response;
import com.ces.hospitalcare.dto.MedicalExaminationDTO;
import com.ces.hospitalcare.dto.UserDTO;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class DoctorResponse {
  private UserDTO doctor;

  private MedicalExaminationDTO medicalExamination;
}
