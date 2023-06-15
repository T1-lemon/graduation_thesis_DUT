package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class NotificationDTO extends BaseDTO {
  private Boolean isRead;

  private AppointmentDTO appointment;
}
