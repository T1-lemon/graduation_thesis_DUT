package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.AppointmentEntity;
import com.ces.hospitalcare.http.response.EmailByDateResponse;
import java.util.List;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface AppointmentRepository extends JpaRepository<AppointmentEntity, Long> {
  List<AppointmentEntity> getAllByPatientIdOrderByModifiedDateDesc(Long patientId);

  @Query(value = "SELECT appointments.* FROM appointments\n"
      + "inner join time_slots\n"
      + "on time_slots.id = appointments.time_slot_id\n"
      + "WHERE appointments.doctor_id = :doctorId and YEARWEEK(time_slots.start_time) = YEARWEEK(NOW());", nativeQuery = true)
  List<AppointmentEntity> getAllAppointmentOfDoctorCurrentWeek(@Param("doctorId") Long doctorId);

  List<AppointmentEntity> getAllByDoctorIdAndPatientIdOrderByModifiedDateDesc(Long doctorId,
      Long patientId);

  AppointmentEntity getByIdAndDoctorId(Long id, Long doctorId);

  List<AppointmentEntity> findAllByStatusAndDoctorIdOrderByModifiedDateDesc(int status,
      Long doctorId, Pageable pageable);

  int countByStatusAndDoctorId(int status, Long doctorId);

  @Query(value = "SELECT * FROM appointments\n"
      + "WHERE appointments.doctor_id = :doctorId and status = 0;", nativeQuery = true)
  List<AppointmentEntity> getAppointmentPendingOfMedicalArchive(@Param("doctorId") Long doctorId);

  @Query(value = "select ts.start_time as startTime, u.email, \n"
      + "u.first_name as firstNamePatient, u.last_name as lastNamePatient, \n"
      + "d.first_name as firstNameDoctor, d.last_name as lastNameDoctor \n"
      + "from appointments as a\n"
      + "inner join users as u\n"
      + "on u.id = a.patient_id\n"
      + "inner join users as d\n"
      + "on d.id = a.doctor_id\n"
      + "inner join time_slots as ts\n"
      + "on a.time_slot_id = ts.id\n"
      + "where ts.start_time like CONCAT(:date, '%') \n"
      + "and status = 1", nativeQuery = true)
  List<EmailByDateResponse> getListEmailByAppointmentDate(@Param("date") String date);
}
