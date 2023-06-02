package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class AppointmentDTO extends BaseDTO {
  private Integer status;

  private UserDTO patient;

  private TimeSlotDTO timeSlot;

  private UserDTO doctor;
}
