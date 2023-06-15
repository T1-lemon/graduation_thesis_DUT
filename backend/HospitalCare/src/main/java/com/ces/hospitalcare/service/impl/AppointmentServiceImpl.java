package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.builder.AppointmentBuilder;
import com.ces.hospitalcare.dto.AppointmentDTO;
import com.ces.hospitalcare.entity.AppointmentEntity;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.entity.TimeSlotEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.request.AppointmentRequest;
import com.ces.hospitalcare.http.request.NotificationRequest;
import com.ces.hospitalcare.http.response.AppointmentResponse;
import com.ces.hospitalcare.http.response.CheckResponse;
import com.ces.hospitalcare.repository.AppointmentRepository;
import com.ces.hospitalcare.repository.MedicalExaminationRepository;
import com.ces.hospitalcare.repository.TimeSlotRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.IAppointmentService;
import com.ces.hospitalcare.service.IEmailService;
import com.ces.hospitalcare.service.INotificationService;
import jakarta.mail.MessagingException;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Pageable;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

@Service
public class AppointmentServiceImpl implements IAppointmentService {
  @Autowired
  private UserRepository userRepository;

  @Autowired
  private AppointmentRepository appointmentRepository;

  @Autowired
  private AppointmentBuilder appointmentBuilder;

  @Autowired
  private TimeSlotRepository timeSlotRepository;

  @Autowired
  private MedicalExaminationRepository medicalExaminationRepository;

  @Autowired
  private ModelMapper mapper;

  @Autowired
  private IEmailService emailService;

  @Autowired
  private INotificationService notificationService;

  @Override
  public int countAppointment() {
    return (int) appointmentRepository.count();
  }

  @Override
  public String cancelAppointmentMedicalArchive(Long doctorId) {
    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.getAppointmentPendingOfMedicalArchive(
        doctorId);
    for (AppointmentEntity entity : listAppointmentEntity) {
      AppointmentDTO appointmentDTO = mapper.map(entity, AppointmentDTO.class);
      appointmentDTO.setStatus(2);
      changeStatusByDoctor(appointmentDTO, doctorId);
    }
    return "cancel successfully";
  }

  public List<AppointmentResponse> getListAppointmentOfPatient() {
    String userEmail = SecurityContextHolder.getContext().getAuthentication().getName();
    UserEntity user = userRepository.findByEmail(userEmail).orElseThrow();

    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.getAllByPatientIdOrderByModifiedDateDesc(
        user.getId());

    List<AppointmentResponse> listAppointmentResponse = new ArrayList<>();

    for (AppointmentEntity entity : listAppointmentEntity) {
      AppointmentResponse appointmentResponse = appointmentBuilder.appointmentResponseBuilder(
          entity);
      listAppointmentResponse.add(appointmentResponse);
    }

    return listAppointmentResponse;
  }

  public List<AppointmentDTO> createAppointmentDTO(
      List<AppointmentEntity> listAppointmentEntity) {
    List<AppointmentDTO> listAppointmentDTO = new ArrayList<>();

    for (AppointmentEntity entity : listAppointmentEntity) {

      AppointmentDTO dto = mapper.map(entity, AppointmentDTO.class);

      listAppointmentDTO.add(dto);
    }

    return listAppointmentDTO;
  }

  @Override
  public int countByStatusAndDoctorId(int status, Long doctorId) {
    return appointmentRepository.countByStatusAndDoctorId(status, doctorId);
  }

  @Override
  public List<AppointmentDTO> getAllAppointmentOfDoctorPageableByStatusAndDoctorId(
      int appointmentStatus, Long doctorId, Pageable pageable) {
    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.findAllByStatusAndDoctorIdOrderByModifiedDateDesc(
        appointmentStatus, doctorId, pageable);

    return createAppointmentDTO(listAppointmentEntity);
  }

  @Override
  public List<AppointmentDTO> getAllAppointmentPageable(Pageable pageable) {

    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.findAll(pageable)
        .getContent();

    return createAppointmentDTO(listAppointmentEntity);
  }

  @Override
  public AppointmentDTO getDetailAppointment(Long appointmentId) {
    AppointmentEntity appointmentEntity = appointmentRepository.getReferenceById(appointmentId);
    return mapper.map(appointmentEntity, AppointmentDTO.class);
  }

  @Override
  public List<AppointmentDTO> getAllByDoctorIdAndPatientId(Long doctorId, Long patientId) {

    return createAppointmentDTO(
        appointmentRepository.getAllByDoctorIdAndPatientIdOrderByModifiedDateDesc(doctorId,
            patientId));
  }

  @Override
  public AppointmentDTO changeStatusByDoctor(AppointmentDTO appointmentChangeStatusDTO,
      Long doctorId) {
    AppointmentEntity appointmentEntityOld = appointmentRepository.getByIdAndDoctorId(
        appointmentChangeStatusDTO.getId(), doctorId);
    appointmentEntityOld.setStatus(appointmentChangeStatusDTO.getStatus());

    AppointmentResponse appointmentResponse = appointmentBuilder.appointmentResponseBuilder(
        appointmentEntityOld);
    String doctorName = appointmentResponse.getFirstNameDoctor() + " "
        + appointmentResponse.getLastNameDoctor();
    String patientName = appointmentResponse.getFirstNamePatient() + " "
        + appointmentResponse.getLastNamePatient();

    String messageBody = "";
    String messageSubject = emailService.messageSubject(doctorName,
        "Appointment Confirmation Notification with ");
    if (appointmentChangeStatusDTO.getStatus() == 1) {
      messageBody = emailService.messageDoctorAcceptAppointmentBody(doctorName, patientName,
          appointmentResponse.getStartTime());
    }

    if (appointmentChangeStatusDTO.getStatus() == 2) {
      messageBody = emailService.messageDoctorCancelAppointmentBody(doctorName, patientName,
          appointmentResponse.getStartTime());
    }

    try {
      emailService.sendEmail(appointmentResponse.getEmailPatient(), messageSubject, messageBody);
    } catch (MessagingException e) {
      throw new RuntimeException(e);
    }

    appointmentRepository.save(appointmentEntityOld);
    //    add notification cho bệnh nhân
    notificationService.addNotification(
        NotificationRequest.builder().appointmentId(appointmentEntityOld.getId()).userId(
            appointmentEntityOld.getPatient().getId()).build());
    return mapper.map(appointmentEntityOld, AppointmentDTO.class);
  }

  @Override
  public AppointmentDTO bookAppointmentByPatient(AppointmentRequest appointmentRequest) {
    UserEntity patient = userRepository.getReferenceById(appointmentRequest.getPatientId());
    UserEntity doctor = userRepository.getReferenceById(appointmentRequest.getDoctorId());
    TimeSlotEntity timeSlotEntity = timeSlotRepository.getReferenceById(
        appointmentRequest.getTimeSlotId());

    AppointmentEntity appointmentEntity = appointmentRepository.save(
        AppointmentEntity.builder().patient(patient)
            .doctor(doctor).timeSlot(timeSlotEntity).build());

    //   add notification cho bác sĩ
    notificationService.addNotification(
        NotificationRequest.builder().appointmentId(appointmentEntity.getId()).userId(
            doctor.getId()).build());

    return mapper.map(appointmentEntity, AppointmentDTO.class);
  }

  @Override
  public CheckResponse checkAppointmentByPatientAndExamination(Long patientId, Long examinationId) {
    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getReferenceById(
        examinationId);
    Long doctorId = medicalExaminationEntity.getDoctor().getId();
    List<AppointmentEntity> appointmentEntityList = appointmentRepository.getAllByDoctorIdAndPatientIdOrderByModifiedDateDesc(
        doctorId, patientId);

    if (appointmentEntityList.size() > 0) {
      return CheckResponse.builder().statusCode(200)
          .message("The patient has an appointment with the doctor").isBooked(true).build();
    }

    return CheckResponse.builder().statusCode(404)
        .message("The patient has never booked an appointment with a doctor").isBooked(false)
        .build();
  }
}
