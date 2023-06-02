import { Avatar } from '@mui/material'
import React from 'react'

type Props = {
  profilePicture: string
  name: string
}

const AdminTableGroupName = (props: Props) => {
  const { profilePicture, name } = props
  return (
    <div>
      <div className='admin__tablegroup--name'>
        <div className='tablegroup__profile--picture'>
          <Avatar src={profilePicture} />
        </div>
        <div className='tablegroup__name'>{name}</div>
      </div>
    </div>
  )
}

export default AdminTableGroupName
