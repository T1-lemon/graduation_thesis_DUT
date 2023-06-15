import CheckOutlinedIcon from '@mui/icons-material/CheckOutlined'
import CloseOutlinedIcon from '@mui/icons-material/CloseOutlined'
import { Button } from '@mui/material'
import {
  DataGrid,
  GridColDef,
  GridColumnGroupHeaderParams,
  GridColumnHeaderParams,
  GridRenderCellParams,
  GridValueGetterParams
} from '@mui/x-data-grid'
import { createFakeServer } from '@mui/x-data-grid-generator'
import React, { useContext } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { toast } from 'react-toastify'

import { IAppointment } from '../../../interface/AppointmentInterface'
import { UserInformation } from '../../../interface/UsersInterface'
import { DispatchType, RootState } from '../../../redux/configStore'
import { changeStatusAppointmentThunk } from '../../../redux/slices/appointmentSlice'
import {
  getNotificationsDoctorThunk,
  readNotificationsDoctorThunk
} from '../../../redux/slices/notificationSlice'
import { changeStatusAppointmentService } from '../../../services/appointmentService'
import { convertVND } from '../../../utils/convertMoney'
import { addHoursToDate, formatDate, getTimeZone } from '../../../utils/date'
import { StatusAppointment } from '../../../utils/statusAppointment'
import {
  DoctorContextType,
  doctorContext
} from '../context/ContextProviderDoctor'

const SERVER_OPTIONS = {
  useCursorPagination: false
}

const { useQuery } = createFakeServer({}, SERVER_OPTIONS)

type Props = {
  appointments: IAppointment[]
  getAllAppointmentPatientForDoctor: () => void
}

export default function TableAppointment(props: Props) {
  const { appointments, getAllAppointmentPatientForDoctor } = props
  const { currentUser } = useSelector((state: RootState) => state.auths)
  const dispatch: DispatchType = useDispatch()
  const { stompDoctor } = useContext(doctorContext) as DoctorContextType

  const changeStatusAppointment = async (
    appointment: IAppointment,
    appointmentStatusChange: number
  ) => {
    await changeStatusAppointmentService(
      appointment.id,
      appointmentStatusChange
    )

    await getAllAppointmentPatientForDoctor()

    await dispatch(readNotificationsDoctorThunk(appointment.id))

    await changeStatusAppointmentSocket(appointment, appointmentStatusChange)
  }

  const changeStatusAppointmentSocket = (
    appointment: IAppointment,
    appointmentStatusChange: number
  ) => {
    if (stompDoctor) {
      let notificationPatient = {
        patientId: appointment.patient.id,
        avatarDoctor: currentUser.profilePicture,
        doctorName: currentUser.firstName + ' ' + currentUser.lastName,
        startTime: appointment.timeSlot.startTime,
        duration: appointment.timeSlot.duration,
        status: appointmentStatusChange,
        appointmentId: appointment.id,
        isRead: false,
        modifiedDate: new Date()
      }
      stompDoctor.send(
        '/app/change-statusAppointment',
        {},
        JSON.stringify(notificationPatient)
      )
    }
  }

  const columns: GridColDef[] = [
    {
      renderHeader: (params: GridColumnHeaderParams) => (
        <strong>{params.colDef.headerName} </strong>
      ),
      field: 'bookingDate',
      headerName: 'Booking Date',
      width: 250
    },
    {
      renderHeader: (params: GridColumnHeaderParams) => (
        <strong>{params.colDef.headerName} </strong>
      ),
      field: 'startTime',
      headerName: 'Start Time',
      sortable: false,
      width: 160
    },
    {
      renderHeader: (params: GridColumnHeaderParams) => (
        <strong>{params.colDef.headerName} </strong>
      ),
      field: 'endTime',
      headerName: 'End Time',
      sortable: false,
      width: 160
    },
    {
      renderHeader: (params: GridColumnHeaderParams) => (
        <strong>{params.colDef.headerName} </strong>
      ),
      field: 'price',
      headerName: 'Price',
      sortable: false,
      width: 120
    },
    {
      renderHeader: (params: GridColumnHeaderParams) => (
        <strong>{params.colDef.headerName} </strong>
      ),
      field: 'status',
      headerName: 'Status',
      sortable: false,
      width: 140,
      renderCell: (params: GridRenderCellParams) => {
        const status = params.value
        if (status === StatusAppointment.Pending)
          return <p className='text__status pending'>Pending</p>
        if (status === StatusAppointment.Approved)
          return <p className='text__status approved'>Approved</p>
        if (status === StatusAppointment.Cancel)
          return <p className='text__status cancel'>Cancel</p>
      }
    },
    {
      renderHeader: (params: GridColumnHeaderParams) => (
        <strong>{params.colDef.headerName} </strong>
      ),
      field: 'action',
      headerName: 'Action',
      sortable: false,
      filterable: false,
      width: 300,
      renderCell: (params: GridRenderCellParams) => {
        const { appointment, status } = params.row
        if (status === StatusAppointment.Pending) {
          return (
            <>
              <Button
                variant='contained'
                className='btn accept'
                startIcon={<CheckOutlinedIcon />}
                onClick={() => {
                  changeStatusAppointment(
                    appointment,
                    StatusAppointment.Approved
                  )
                  toast.success('Appointment Approved')
                }}
              >
                Approve
              </Button>
              <Button
                variant='contained'
                className='btn cancel'
                startIcon={<CloseOutlinedIcon />}
                onClick={() => {
                  changeStatusAppointment(appointment, StatusAppointment.Cancel)
                  toast.success('Appointment Canceled')
                }}
              >
                Cancel
              </Button>
            </>
          )
        }
      }
    }
  ]

  const rows = appointments.map((item, index) => {
    const { timeSlot } = item

    const rowItem = {
      id: index,
      appointmentId: item.id,
      bookingDate: formatDate(new Date(timeSlot.startTime)),
      startTime: getTimeZone(timeSlot.startTime),
      endTime: getTimeZone(
        addHoursToDate(new Date(timeSlot.startTime), timeSlot.duration)
      ),
      price: convertVND.format(timeSlot.medicalExamination.examinationPrice),
      status: item.status,
      appointment: item
    }

    return rowItem
  })

  // loading
  const [page, setPage] = React.useState(0)

  const queryOptions = React.useMemo(
    () => ({
      page
    }),
    [page]
  )

  const { isLoading } = useQuery(queryOptions)

  return (
    <div className='container__table__appointment'>
      <DataGrid
        sx={{
          boxShadow: 2,
          '& .MuiDataGrid-cell:focus': {
            outline: 'none'
          },
          '& .MuiDataGrid-columnHeader,.MuiDataGrid-cell': {
            padding: '0px 26px'
          }
        }}
        rows={rows}
        rowHeight={70}
        disableSelectionOnClick
        columns={columns}
        pageSize={6}
        loading={isLoading}
        pagination
        onPageChange={(newPage) => setPage(newPage)}
      />
    </div>
  )
}
