import { TableRow } from '@mui/material'

import {
  AdminTableColumn,
  TablePatient
} from '../../../interface/AdminTableInterface'
import { TableCellProfile } from '../../../themes/profileStyle'
import AdminGroupBtnAction from './AdminGroupBtnAction'
import AdminTableGroupName from './AdminTableGroupName'

interface Props {
  columns: AdminTableColumn[]
  row: TablePatient
}

const AdminPatientTableCell = (props: Props) => {
  const { columns, row } = props
  return (
    <TableRow hover role='checkbox' tabIndex={-1}>
      {columns.map((column: AdminTableColumn) => {
        if (column.id === 'actions') {
          return (
            <TableCellProfile key={column.id}>
              <AdminGroupBtnAction patientId={row.id} />
            </TableCellProfile>
          )
        }
        const columnId = column.id as keyof typeof row
        const value = row[columnId]
        if (value === row.patientName) {
          return (
            <TableCellProfile key={column.id} align={column.align}>
              <AdminTableGroupName
                profilePicture={row.profilePicturePatient}
                name={row.patientName}
              />
            </TableCellProfile>
          )
        }
        return (
          <TableCellProfile key={column.id} align={column.align}>
            {value}
          </TableCellProfile>
        )
      })}
    </TableRow>
  )
}

export default AdminPatientTableCell
