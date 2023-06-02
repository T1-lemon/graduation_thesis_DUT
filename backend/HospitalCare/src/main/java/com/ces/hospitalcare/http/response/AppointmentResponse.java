package com.ces.hospitalcare.http.response;
import java.util.Date;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class AppointmentResponse {
  private Long patientId;

  private Long appointmentId;

  private Long doctorId;

  private String profilePictureDoctor;

  private String firstNameDoctor;

  private String lastNameDoctor;

  private Boolean genderDoctor;

  private Long examinationPrice;

  private Date startTime;

  private Integer duration;

  private String nameDepartment;

  private Integer status;
}
