package com.ces.hospitalcare.builder;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.http.request.MedicalExaminationRequest;
import com.ces.hospitalcare.repository.DepartmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class MedicalExaminationBuilder {
  @Autowired
  private DepartmentRepository departmentRepository;

  public MedicalExaminationEntity medicalUpdateBuild(MedicalExaminationRequest medicalUpdate,
      MedicalExaminationEntity medicalOld) {
    medicalOld.setExaminationPrice(medicalUpdate.getExaminationPrice());
    medicalOld.setTitle(medicalUpdate.getTitle());
    medicalOld.setDescription(medicalUpdate.getDescription());
    medicalOld.setShortDescription(medicalUpdate.getShortDescription());

    return medicalOld;
  }
}
