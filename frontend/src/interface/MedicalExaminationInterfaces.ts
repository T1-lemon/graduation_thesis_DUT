import { IDepartment } from './Department'
import { ITimeSlotResponse } from './TimeSlotInterfaces'

export interface IMedicalExamination {
  id: number
  createdBy: null
  modifiedBy: null
  createdDate: null
  modidiedDate: null
  examinationPrice: number
  title: string
  shortDescription: string
  description: string
  examinationObject: number
  image: string
  department: IDepartment
}

export interface IMedicalExaminationFilter {
  category: string[]
  minPrice: number
  maxPrice: number
}

export interface IMedicalExaminationTime {
  medicalExamination: IMedicalExamination
  listTimeSlot: ITimeSlotResponse[]
}
