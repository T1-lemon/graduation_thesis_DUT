package com.ces.hospitalcare.http.response;
import com.ces.hospitalcare.dto.MedicalExaminationDTO;
import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class MedicalExaminationResponse {
  private MedicalExaminationDTO medicalExamination;

  private List<TimeSlotResponse> listTimeSlot = new ArrayList<>();
}
