package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.builder.AppointmentBuilder;
import com.ces.hospitalcare.dto.NotificationDTO;
import com.ces.hospitalcare.entity.AppointmentEntity;
import com.ces.hospitalcare.entity.NotificationEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.request.NotificationRequest;
import com.ces.hospitalcare.http.response.AppointmentResponse;
import com.ces.hospitalcare.http.response.NotificationPatientResponse;
import com.ces.hospitalcare.repository.AppointmentRepository;
import com.ces.hospitalcare.repository.NotificationRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.INotificationService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class NotificationServiceImpl implements INotificationService {
  @Autowired
  private NotificationRepository notificationRepository;

  @Autowired
  private AppointmentBuilder appointmentBuilder;

  @Autowired
  private ModelMapper mapper;

  @Autowired
  private UserRepository userRepository;

  @Autowired
  private AppointmentRepository appointmentRepository;

  @Override
  public List<NotificationDTO> getNotificationsDoctor(Long doctorId) {
    List<NotificationEntity> notificationEntityList = notificationRepository.getNotificationsUserInTwoMonth(
        doctorId);

    List<NotificationDTO> notificationDTOList = new ArrayList<>();
    for (NotificationEntity entity : notificationEntityList) {
      NotificationDTO notificationDTO = mapper.map(entity, NotificationDTO.class);
      notificationDTOList.add(notificationDTO);
    }

    return notificationDTOList;
  }

  @Override
  public List<NotificationPatientResponse> getNotificationsPatient(Long patientId) {
    List<NotificationEntity> notificationEntityList = notificationRepository.getNotificationsUserInTwoMonth(
        patientId);

    List<NotificationPatientResponse> notificationResponseList = new ArrayList<>();
    for (NotificationEntity entity : notificationEntityList) {
      AppointmentResponse appointmentResponse = appointmentBuilder.appointmentResponseBuilder(
          entity.getAppointment());

      notificationResponseList.add(
          NotificationPatientResponse.builder().inforNotification(appointmentResponse)
              .isRead(entity.getIsRead())
              .build());
    }

    return notificationResponseList;
  }

  @Override
  public NotificationDTO addNotification(NotificationRequest notificationRequest) {
    UserEntity userEntity = userRepository.getReferenceById(notificationRequest.getUserId());
    AppointmentEntity appointmentEntity = appointmentRepository.getReferenceById(
        notificationRequest.getAppointmentId());
    NotificationEntity notificationEntity = notificationRepository.save(
        NotificationEntity.builder().appointment(appointmentEntity).user(userEntity).build());
    return mapper.map(notificationEntity, NotificationDTO.class);
  }

  @Override
  public NotificationDTO changeReadNotification(NotificationRequest notificationRequest) {
    NotificationEntity notificationEntity = notificationRepository.findByAppointmentIdAndUserId(
        notificationRequest.getAppointmentId(), notificationRequest.getUserId());
    notificationEntity.setIsRead(true);

    notificationRepository.save(notificationEntity);

    return mapper.map(notificationEntity, NotificationDTO.class);
  }
}
