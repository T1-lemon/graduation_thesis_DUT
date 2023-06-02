package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.MedicalExaminationDTO;
import com.ces.hospitalcare.http.request.MedicalExaminationRequest;
import com.ces.hospitalcare.http.response.MedicalExaminationResponse;
import java.util.List;

public interface IMedicalExaminationService {
  List<MedicalExaminationResponse> getAllMedicalExamination();

  MedicalExaminationResponse getDetailMedicalExamination(Long id);

  List<MedicalExaminationResponse> filterMedicalExaminationByCategoryAndPrice(
      Long startPrice, Long endPrice, String[] categories);

  MedicalExaminationDTO archiveMedicalExamination(MedicalExaminationDTO medicalExaminationDTO);

  MedicalExaminationDTO addMedicalExamination(MedicalExaminationRequest medicalExaminationRequest);

  MedicalExaminationDTO updateMedicalExamination(
      MedicalExaminationRequest medicalExaminationRequest, Long medicalId);
}
