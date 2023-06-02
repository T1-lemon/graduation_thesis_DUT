package com.ces.hospitalcare.builder;
import com.ces.hospitalcare.entity.AppointmentEntity;
import com.ces.hospitalcare.entity.DepartmentEntity;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.entity.TimeSlotEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.response.AppointmentResponse;
import org.springframework.stereotype.Component;

@Component
public class AppointmentBuilder {
  public AppointmentResponse appointmentResponseBuilder(AppointmentEntity appointmentEntity) {
    UserEntity doctor = appointmentEntity.getDoctor();
    UserEntity patient = appointmentEntity.getPatient();
    TimeSlotEntity timeSlotAppointment = appointmentEntity.getTimeSlot();
    MedicalExaminationEntity medicalExaminationDoctor = doctor.getMedicalExaminationEntity();
    DepartmentEntity departmentDoctor = medicalExaminationDoctor.getDepartment();
    return AppointmentResponse.builder()
        .appointmentId(appointmentEntity.getId())
        .patientId(patient.getId())
        .doctorId(doctor.getId())
        .firstNameDoctor(doctor.getFirstName())
        .lastNameDoctor(doctor.getLastName())
        .genderDoctor(doctor.getGender())
        .profilePictureDoctor(doctor.getProfilePicture())
        .nameDepartment(departmentDoctor.getName())
        .startTime(timeSlotAppointment.getStartTime())
        .duration(timeSlotAppointment.getDuration())
        .examinationPrice(medicalExaminationDoctor.getExaminationPrice())
        .status(appointmentEntity.getStatus())
        .build();
  }
}
