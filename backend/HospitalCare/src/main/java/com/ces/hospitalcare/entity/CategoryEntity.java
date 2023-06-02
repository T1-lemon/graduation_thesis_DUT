package com.ces.hospitalcare.entity;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.ManyToMany;
import jakarta.persistence.Table;
import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "Categories")
public class CategoryEntity extends BaseEntity {
  @Column
  private String name;

  @ManyToMany(mappedBy = "listCategoryEntity")
  private List<MedicalExaminationEntity> listMedicalExaminationEntity = new ArrayList<>();
}
