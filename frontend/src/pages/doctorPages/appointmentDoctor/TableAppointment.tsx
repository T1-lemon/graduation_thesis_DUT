import {
  DataGrid,
  GridColDef,
  GridColumnGroupHeaderParams,
  GridColumnHeaderParams,
  GridRenderCellParams,
  GridValueGetterParams
} from '@mui/x-data-grid'
import { createFakeServer } from '@mui/x-data-grid-generator'
import * as React from 'react'
import { useSelector } from 'react-redux'

import { IAppointment } from '../../../interface/AppointmentInterface'
import { UserInformation } from '../../../interface/UsersInterface'
import { RootState } from '../../../redux/configStore'
import { convertVND } from '../../../utils/convertMoney'
import { addHoursToDate, formatDate, getTimeZone } from '../../../utils/date'
import { StatusAppointment } from '../../../utils/statusAppointment'

const SERVER_OPTIONS = {
  useCursorPagination: false
}

const { useQuery } = createFakeServer({}, SERVER_OPTIONS)

type Props = {
  appointments: IAppointment[]
}

export default function TableAppointment(props: Props) {
  const appointments = props.appointments
  const { currentUser } = useSelector((state: RootState) => state.auths)

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
      width: 180
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
    }
  ]

  const rows = appointments.map((item, index) => {
    const { timeSlot } = item

    const rowItem = {
      id: index,
      doctor: currentUser,
      bookingDate: formatDate(new Date(timeSlot.startTime)),
      startTime: getTimeZone(timeSlot.startTime),
      endTime: getTimeZone(
        addHoursToDate(new Date(timeSlot.startTime), timeSlot.duration)
      ),
      price: convertVND.format(timeSlot.medicalExamination.examinationPrice),
      status: item.status
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
