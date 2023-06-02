package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface MedicalExaminationRepository extends
    JpaRepository<MedicalExaminationEntity, Long> {
  @Query(
      value =
          "select  medical_examinations.* from  medical_examinations  \n"
              + "inner join categories_medical_examinations\n"
              + "on medical_examinations.id = categories_medical_examinations.examination_id \n"
              + "inner join categories\n"
              + "on categories_medical_examinations.category_id = categories.id \n"
              + "where examination_price between :minPrice and :maxPrice && categories.name in (:categories) \n"
              + "group by medical_examinations.id;",
      nativeQuery = true)
  List<MedicalExaminationEntity> filterMedicalExaminationByCategoryAndPrice(
      @Param("minPrice") long minPrice,
      @Param("maxPrice") long maxPrice,
      @Param("categories") String[] categories);

  @Query(
      value =
          "select  * from  medical_examinations where examination_price between :minPrice and :maxPrice",
      nativeQuery = true)
  List<MedicalExaminationEntity> filterMedicalExaminationByPrice(
      @Param("minPrice") long minPrice, @Param("maxPrice") long maxPrice);

  @Query(
      value =
          "select  medical_examinations.* from  medical_examinations  \n"
              + "inner join categories_medical_examinations\n"
              + "on medical_examinations.id = categories_medical_examinations.examination_id \n"
              + "inner join categories\n"
              + "on categories_medical_examinations.category_id = categories.id \n"
              + "where  categories.name in (:categories)\n"
              + "group by medical_examinations.id;",
      nativeQuery = true)
  List<MedicalExaminationEntity> filterMedicalExaminationByCategory(
      @Param("categories") String[] categories);

  MedicalExaminationEntity getByDoctorId(Long doctorId);

  List<MedicalExaminationEntity> getAllByStatusArchive(Integer statusArchive);

  List<MedicalExaminationEntity> getAllByDepartmentId(Long departmentId);
}
