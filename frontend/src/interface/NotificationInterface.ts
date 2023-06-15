import { AppointmentInforInterface, IAppointment } from './AppointmentInterface';


export interface INotificationPatientSocket {
  patientId: number
  avatarDoctor: string
  startTime: string
  duration: number
  doctorName: string
  status: number
  appointmentId: number
  isRead?: boolean
  modifiedDate: Date
}

export interface INotificationPatientApi {
  isRead: boolean
  inforNotification: AppointmentInforInterface
}

export interface INotificationDoctorApi {
  id: number
  isRead: boolean
  appointment: IAppointment
}

export interface INotificationDoctorSocket {
  patientId?: number
  doctorId?: number
  avatarPatient?: string
  startTime: string
  duration: number
  patientName?: string
  isRead: boolean
  createdDate: Date
}