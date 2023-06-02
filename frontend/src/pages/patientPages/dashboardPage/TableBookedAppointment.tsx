import SearchIcon from '@mui/icons-material/Search'
import {
  Avatar,
  Paper,
  Table,
  TableBody,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow
} from '@mui/material'
import { IconButton, InputAdornment } from '@mui/material'
import { useEffect, useState } from 'react'

import { ProgressListener } from '../../../components/Progress'
import { useAppDispatch, useAppSelector } from '../../../redux/hooks'
import { getListAppointment } from '../../../redux/thunk/appointmentThunk'
import {
  TableCellProfile,
  TextFieldProfile
} from '../../../themes/profileStyle'
import { convertVND } from '../../../utils/convertMoney'
import { addHoursToDate, formatDate, getTimeZone } from '../../../utils/date'
import { StatusAppointment } from '../../../utils/statusAppointment'

interface Column {
  id: 'doctorName' | 'apptDate' | 'startTime' | 'endTime' | 'price' | 'status'
  label: string
  minWidth?: number
  align?: 'right'
}

const columns: readonly Column[] = [
  { id: 'doctorName', label: 'Doctor Name', minWidth: 170 },
  { id: 'apptDate', label: 'Appt Date', minWidth: 100 },
  { id: 'startTime', label: 'Start Time', minWidth: 100 },
  { id: 'endTime', label: 'End Time', minWidth: 100 },
  { id: 'price', label: 'Price', minWidth: 100 },
  { id: 'status', label: 'Status', minWidth: 100 }
]

interface TableAppointment {
  doctorName: string
  profilePicture: string
  departmentName: string
  apptDate: string
  startTime: string
  endTime: string
  price: string
  status: number
}

function createDataTableAppointment(
  doctorName: string,
  profilePicture: string,
  departmentName: string,
  apptDate: string,
  startTime: string,
  endTime: string,
  price: string,
  status: number
): TableAppointment {
  return {
    doctorName,
    profilePicture,
    departmentName,
    apptDate,
    startTime,
    endTime,
    price,
    status
  }
}

const rowsPerPage = 6

function TableBookedAppointment() {
  const [page, setPage] = useState(0)
  const [appointmentRows, setAppointmentRows] = useState<TableAppointment[]>([])
  const [appointmentRowsSearched, setAppointmentRowsSearched] = useState<
    TableAppointment[]
  >([])
  const dispatch = useAppDispatch()

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage)
  }

  const { listAppointments } = useAppSelector((state) => state.appointments)
  const { isCheckInitialStatus } = useAppSelector((state) => state.auths)
  const [searchTerm, setSearchTerm] = useState<string>('')

  const handleChangeSearchBox = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSearchTerm(event.target.value)
  }

  const handleClickSearchBox = () => {
    const searchAppointments = appointmentRows.filter((item) =>
      item.doctorName.includes(searchTerm)
    )
    setAppointmentRowsSearched(searchAppointments)
  }

  useEffect(() => {
    if (!isCheckInitialStatus) {
      const fetchApiGetListAppointment = async () => {
        ProgressListener.emit('start')
        await dispatch(getListAppointment())
        ProgressListener.emit('stop')
      }
      fetchApiGetListAppointment()
      return
    }
  }, [])

  useEffect(() => {
    const appointments = listAppointments.map((appointment) => {
      const date = new Date(appointment.startTime)
      return createDataTableAppointment(
        `Dr. ${appointment.firstNameDoctor} ${appointment.lastNameDoctor}`,
        `${appointment.profilePictureDoctor}`,
        `${appointment.nameDepartment}`,
        `${formatDate(date)}`,
        `${getTimeZone(appointment.startTime)}`,
        `${getTimeZone(addHoursToDate(date, appointment.duration))}`,
        `${convertVND.format(appointment.examinationPrice)}`,
        appointment.status
      )
    })
    setAppointmentRows(appointments)
    setAppointmentRowsSearched(appointments)
  }, [listAppointments])

  return (
    <>
      <TextFieldProfile
        id='search'
        type='search'
        size='small'
        value={searchTerm}
        onChange={handleChangeSearchBox}
        InputProps={{
          endAdornment: (
            <InputAdornment position='end'>
              <IconButton onClick={handleClickSearchBox}>
                <SearchIcon />
              </IconButton>
            </InputAdornment>
          )
        }}
      />
      <div className='table__container'>
        <TableContainer component={Paper}>
          <Table sx={{ minWidth: 650 }} aria-label='simple table'>
            <TableHead>
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
              {appointmentRowsSearched
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row, index) => {
                  return (
                    <TableRow hover role='checkbox' tabIndex={-1} key={index}>
                      {columns.map((column) => {
                        const value = row[column.id]
                        return (
                          <TableCellProfile
                            key={column.id}
                            align={column.align}
                          >
                            {value === row.doctorName ? (
                              <>
                                <div className='doctor__group--name'>
                                  <div className='doctor__profile--picture'>
                                    <Avatar src={row.profilePicture} />
                                  </div>
                                  <div className='doctor__name'>
                                    {row.doctorName}
                                    <span>{row.departmentName}</span>
                                  </div>
                                </div>
                              </>
                            ) : value === row.status ? (
                              <>
                                <RenderStatus status={row.status} />
                              </>
                            ) : (
                              value
                            )}
                          </TableCellProfile>
                        )
                      })}
                    </TableRow>
                  )
                })}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[6]}
          component='div'
          count={appointmentRows.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
        />
      </div>
    </>
  )
}

export default TableBookedAppointment

type IProps = {
  status?: number
}

export const RenderStatus = (props: IProps) => {
  const { status } = props
  if (status === StatusAppointment.Pending)
    return <p className='apt__txt--status txt--pending'>Pending</p>
  if (status === StatusAppointment.Approved)
    return <p className='apt__txt--status txt--approved'>Approved</p>
  if (status === StatusAppointment.Cancel)
    return <p className='apt__txt--status txt--cancel'>Cancel</p>
  return <></>
}
