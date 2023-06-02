package com.ces.hospitalcare.entity;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.JoinTable;
import jakarta.persistence.ManyToMany;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Entity
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "MedicalExaminations")
public class MedicalExaminationEntity extends BaseEntity {
  @Column
  private Long examinationPrice;

  @Column
  private String title;

  @Column(columnDefinition = "TEXT")
  private String shortDescription;

  @Column(columnDefinition = "TEXT")
  private String description;

  @Column
  private String image;

  @Column
  @Builder.Default
  private Integer statusArchive = 0;

  @OneToMany(mappedBy = "medicalExamination")
  private List<TimeSlotEntity> timeSlots = new ArrayList<>();

  @OneToMany(mappedBy = "medicalExamination")
  private List<FeedbackEntity> feedBacks = new ArrayList<>();

  @OneToOne
  @JoinColumn(name = "doctorId")
  private UserEntity doctor;

  @ManyToOne
  @JoinColumn(name = "departmentId")
  private DepartmentEntity department;

  @ManyToMany
  @JoinTable(
      name = "CategoriesMedicalExaminations",
      joinColumns = @JoinColumn(name = "examinationId"),
      inverseJoinColumns = @JoinColumn(name = "categoryId"))
  private List<CategoryEntity> listCategoryEntity = new ArrayList<>();
}
