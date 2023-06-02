import {
  Paper,
  Table,
  TableBody,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow
} from '@mui/material'
import { useEffect, useState } from 'react'

import {
  AdminTableColumn,
  TableAppointment
} from '../../interface/AdminTableInterface'
import { getAllAppointmentService } from '../../services/adminServices/adminAppointmentServices'
import { TableCellProfile } from '../../themes/profileStyle'
import { addHoursToDate, formatDate, getTimeZone } from '../../utils/date'
import AdminBreadCrumb from './adminBreadcrumb/AdminBreadCrumb'
import AdminAppointmentTableCell from './adminTable/AdminAppointmentTableCell'

const columns: AdminTableColumn[] = [
  { id: 'doctorName', label: 'Doctors', minWidth: 170 },
  { id: 'patientName', label: 'Patients', minWidth: 170 },
  { id: 'apptDate', label: 'Appointment Date', minWidth: 170 },
  { id: 'status', label: 'Status', minWidth: 10 },
  { id: 'actions', label: 'Actions', minWidth: 70, align: 'right' }
]

function createDataTableAppointment(
  id: number,
  doctorName: string,
  profilePictureDoctor: string,
  patientName: string,
  profilePicturePatient: string,
  apptDate: string,
  startTime: string,
  endTime: string,
  status: number
): TableAppointment {
  return {
    id,
    doctorName,
    profilePictureDoctor,
    patientName,
    profilePicturePatient,
    apptDate,
    startTime,
    endTime,
    status
  }
}

const rowsPerPage = 5

const AdminAppointmentPage = () => {
  const [page, setPage] = useState(0)
  const [totalPage, setTotalPage] = useState<number>(0)
  const [rowsAppointment, setRowsAppointment] = useState<any[]>([])
  const [rowsRenderAppt, setRowsRenderAppt] = useState<any[]>([])

  const fetchGetAllAppointmentApi = async (pageIndex: number) => {
    const response = await getAllAppointmentService(pageIndex, rowsPerPage)
    setTotalPage(response.totalPage)
    const appointments = response.listAppointmentResult.map(
      (appointment: any) => {
        const { id, status, doctor, patient, timeSlot } = appointment
        const date = new Date(timeSlot.startTime)
        return createDataTableAppointment(
          id,
          `Dr ${doctor.firstName} ${doctor.lastName}`,
          doctor.profilePicture,
          `${patient.firstName} ${patient.lastName}`,
          patient.profilePicture,
          `${formatDate(date)}`,
          `${getTimeZone(timeSlot.startTime)}`,
          `${getTimeZone(addHoursToDate(date, timeSlot.duration))}`,
          status
        )
      }
    )
    if (rowsAppointment.length === 0) {
      setRowsAppointment(appointments)
      return
    }

    const rowsAppointmentClone = rowsAppointment
    appointments.forEach((appointment: any) => {
      if (
        appointment.id >
        rowsAppointmentClone[rowsAppointmentClone.length - 1].id
      ) {
        rowsAppointmentClone.push(appointment)
      }
      if (
        appointment.id <=
        rowsAppointmentClone[rowsAppointmentClone.length - 1].id
      ) {
        rowsAppointmentClone.map((rowAppointment) => {
          if (appointment.id === rowAppointment.id) {
            return appointment
          }
          return rowAppointment
        })
      }
    })
    setRowsAppointment(rowsAppointmentClone)
  }

  useEffect(() => {
    fetchGetAllAppointmentApi(1)
  }, [])

  useEffect(() => {
    setRowsRenderAppt(rowsAppointment)
  }, [rowsAppointment])

  const handleChangePage = async (event: unknown, newPage: number) => {
    await fetchGetAllAppointmentApi(newPage + 1)
    setPage(newPage)
  }

  return (
    <div className='admin__appointment'>
      <div className='admin__appointment--header'>
        <AdminBreadCrumb breadcrumbTitle='Appointments' />
      </div>
      <div className='admin__appointment--body'>
        <div className='admin__appointment--table'>
          <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label='simple table'>
              <TableHead className='admin__table--head'>
                <TableRow>
                  {columns.map((column, index) => (
                    <TableCellProfile
                      key={index}
                      align={column.align}
                      style={{ minWidth: column.minWidth }}
                    >
                      {column.label}
                    </TableCellProfile>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {rowsRenderAppt ? (
                  rowsRenderAppt
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((row, index) => {
                      return (
                        <AdminAppointmentTableCell
                          columns={columns}
                          row={row}
                          key={index}
                        />
                      )
                    })
                ) : (
                  <></>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[6]}
            component='div'
            count={totalPage * rowsPerPage}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
          />
        </div>
      </div>
    </div>
  )
}

export default AdminAppointmentPage
