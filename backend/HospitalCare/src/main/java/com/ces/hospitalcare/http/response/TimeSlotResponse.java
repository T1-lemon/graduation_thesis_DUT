package com.ces.hospitalcare.http.response;
import com.ces.hospitalcare.dto.TimeSlotDTO;
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
public class TimeSlotResponse {
  private TimeSlotDTO timeSlotDTO;

  private Long doctorId;

  private Long appointmentId;
}
